"""정답표를 기준으로 실제 RAG를 평가하고 JSON/CSV 보고서를 저장합니다."""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import (
    ROOT, UNKNOWN, answer_from_documents, build_store, error_message,
    file_manifest, load_documents, normalize, split_documents,
)


class ExpectedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=8)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    reference_answer: str = Field(min_length=1)
    # 각 그룹에서 최소 한 단어가 있으면 통과합니다. 동의어를 같은 그룹에 적습니다.
    required_terms: list[list[str]]
    forbidden_terms: list[str] = Field(default_factory=list)
    expected_evidence: list[ExpectedEvidence]

    @model_validator(mode="after")
    def check_gold(self):
        if self.answerable and (not self.required_terms or not self.expected_evidence):
            raise ValueError("답변 가능한 질문에는 필수 단어와 원문 근거가 필요합니다.")
        if not self.answerable and (self.required_terms or self.expected_evidence):
            raise ValueError("답변 불가능한 질문의 필수 단어와 근거는 비워 두세요.")
        if any(not group or any(not word.strip() for word in group) for group in self.required_terms):
            raise ValueError("필수 단어 그룹은 비워 둘 수 없습니다.")
        if any(not word.strip() for word in self.forbidden_terms):
            raise ValueError("금지 단어에는 빈 문자열을 넣을 수 없습니다.")
        return self


def read_cases(path: Path) -> list[EvaluationCase]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("평가 케이스는 비어 있지 않은 JSON 배열이어야 합니다.")
    cases = [EvaluationCase.model_validate(item) for item in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("평가 케이스의 id는 중복할 수 없습니다.")
    return cases


def evidence_matches(evidence: ExpectedEvidence, document: Document) -> bool:
    # 페이지 번호뿐 아니라 정답의 원문까지 검색했는지 확인합니다.
    return (
        document.metadata.get("source") == evidence.source
        and document.metadata.get("page") == evidence.page
        and normalize(evidence.quote) in normalize(document.page_content)
    )


def check_gold(cases: list[EvaluationCase], documents: list[Document]) -> None:
    # 정답표의 페이지나 인용문이 틀리면 API를 호출하기 전에 중단합니다.
    for case in cases:
        for evidence in case.expected_evidence:
            if not any(evidence_matches(evidence, doc) for doc in documents):
                raise ValueError(f"{case.id}: 정답 근거를 원문에서 확인할 수 없습니다: {evidence.source}, {evidence.page}쪽")


def score_case(case: EvaluationCase, retrieved: list[Document], response: dict) -> dict:
    answer = response["answer"]
    sources = response.get("sources", [])
    refused = normalize(answer) == normalize(UNKNOWN) and not sources
    valid_sources = [
        bool(normalize(source.get("quote", ""))) and any(
            doc.metadata.get("source") == source.get("source")
            and doc.metadata.get("page") == source.get("page")
            and normalize(source["quote"]) in normalize(doc.page_content)
            for doc in retrieved
        )
        for source in sources
    ]
    if case.answerable:
        retrieval_ok = all(any(evidence_matches(e, doc) for doc in retrieved)
                           for e in case.expected_evidence)
        terms_ok = all(any(normalize(term) in normalize(answer) for term in group)
                       for group in case.required_terms)
        terms_ok = terms_ok and not any(normalize(term) in normalize(answer)
                                       for term in case.forbidden_terms)
        # 인용이 실제 존재하는지, 정답 근거의 원문과 겹치는지도 확인합니다.
        expected_cited = all(any(
            source.get("source") == e.source and source.get("page") == e.page
            and (normalize(e.quote) in normalize(source.get("quote", ""))
                 or normalize(source.get("quote", "")) in normalize(e.quote))
            for source in sources if len(normalize(source.get("quote", ""))) >= 8
        ) for e in case.expected_evidence)
        citation_ok = bool(sources) and all(valid_sources) and expected_cited
        checks = {"retrieval_pass": retrieval_ok, "answer_terms_pass": terms_ok and not refused,
                  "citation_pass": citation_ok, "refusal_pass": None}
    else:
        checks = {"retrieval_pass": None, "answer_terms_pass": None,
                  "citation_pass": None, "refusal_pass": refused}
    failures = [name for name, passed in checks.items() if passed is False]
    return {
        "id": case.id, "question": case.question, "answerable": case.answerable,
        "reference_answer": case.reference_answer, "answer": answer,
        "expected_evidence": [e.model_dump() for e in case.expected_evidence],
        "sources": sources,
        "retrieved": [{**doc.metadata, "text": doc.page_content} for doc in retrieved],
        **checks, "automatic_pass": not failures, "failure_checks": failures,
        # 단어 일치만으로 의미의 정확성을 보장할 수 없어 사람이 확인할 칸을 남깁니다.
        "manual_review": "pending", "manual_notes": "",
    }


def summarize(rows: list[dict]) -> dict:
    summary = {"total": len(rows), "automatic_pass_count": sum(r["automatic_pass"] for r in rows),
               "error_count": sum(bool(r.get("error")) for r in rows)}
    for name in ("retrieval_pass", "answer_terms_pass", "citation_pass", "refusal_pass"):
        applicable = [r for r in rows if r.get(name) is not None]
        passed = sum(r[name] for r in applicable)
        summary[name] = {"passed": passed, "total": len(applicable),
                         "rate": passed / len(applicable) if applicable else None}
    return summary


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 엑셀에서 한글을 읽을 수 있도록 UTF-8 BOM이 있는 CSV도 저장합니다.
    fields = ["id", "question", "reference_answer", "answer", "retrieval_pass",
              "answer_terms_pass", "citation_pass", "refusal_pass", "automatic_pass",
              "failure_checks", "error", "manual_review", "manual_notes"]
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])


def main() -> int:
    parser = argparse.ArgumentParser(description="문서 RAG의 검색·답변·근거를 검증합니다.")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/results/latest.json")
    parser.add_argument("--check-only", action="store_true", help="API 호출 없이 정답표와 원문 확인")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Python 3.11로 실행해 주세요.")
    cases = read_cases(args.cases)
    manifest = file_manifest()
    documents, notices = load_documents(manifest)
    check_gold(cases, documents)
    print(f"정답표 검증: {len(cases)}건, 원문 근거와 일치")
    if args.check_only:
        return 0
    key = (dotenv_values(ROOT / ".env", encoding="utf-8-sig").get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(".env의 OPENAI_API_KEY를 설정해 주세요.")
    chunks = split_documents(documents)
    store = build_store(chunks, key)
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            retrieved = store.as_retriever(search_kwargs={"k": 6}).invoke(case.question)
            response = answer_from_documents(retrieved, case.question, key)
            row = score_case(case, retrieved, response)
        except Exception as error:
            # 오류가 난 질문도 평가에서 제외하지 않고 실패로 기록합니다.
            row = {"id": case.id, "question": case.question,
                   "reference_answer": case.reference_answer, "answer": "",
                   "automatic_pass": False, "error": error_message(error),
                   "failure_checks": ["execution_error"], "manual_review": "pending"}
            for name in ("retrieval_pass", "answer_terms_pass", "citation_pass"):
                row[name] = False if case.answerable else None
            row["refusal_pass"] = None if case.answerable else False
        row["seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        print(f"{case.id}: {'PASS' if row['automatic_pass'] else 'FAIL'}")
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "models": {"embedding": "text-embedding-3-small", "answer": "gpt-4o-mini"},
        "settings": {"chunk_size": 1000, "chunk_overlap": 150, "k": 6},
        "packages": {p: version(p) for p in ("langchain", "langchain-openai", "pypdf")},
        "data_manifest": manifest,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "extraction_notices": notices, "chunk_count": len(chunks),
        "summary": summarize(rows), "results": rows,
        "limitations": "자동 판정은 단어와 원문의 일치를 확인합니다. 의미·조건·예외는 사람이 검토해야 합니다.",
    }
    write_report(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"보고서: {args.output} / {args.output.with_suffix('.csv')}")
    return 0 if all(row["automatic_pass"] for row in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"검증 실패: {error_message(error)}", file=sys.stderr)
        raise SystemExit(2)
