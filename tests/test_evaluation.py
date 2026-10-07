"""API 호출 없이 평가 로직의 성공·실패 판정을 검사합니다."""

import unittest

from langchain_core.documents import Document

from app import Evidence, GroundedAnswer, UNKNOWN, validate_answer
from evaluate import EvaluationCase, check_gold, score_case, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.quote = "근무지외 국내출장 운임, 식비, 숙박비, 일비"
        self.doc = Document(page_content=self.quote,
                            metadata={"source": "기준.pdf", "page": 8})
        self.case = EvaluationCase(
            id="expenses", question="여비 항목은?", answerable=True,
            reference_answer="운임, 식비, 숙박비, 일비",
            required_terms=[["운임"], ["식비"], ["숙박비"], ["일비"]],
            forbidden_terms=["무조건"],
            expected_evidence=[{"source": "기준.pdf", "page": 8, "quote": self.quote}],
        )
        self.response = {"answer": "운임, 식비, 숙박비, 일비가 포함됩니다.",
                         "sources": [{**self.doc.metadata, "quote": self.quote}]}

    def test_valid_answer(self):
        result = score_case(self.case, [self.doc], self.response)
        self.assertTrue(result["automatic_pass"])
        self.assertEqual(result["manual_review"], "pending")

    def test_missing_retrieval(self):
        result = score_case(self.case, [], self.response)
        self.assertFalse(result["retrieval_pass"])
        self.assertFalse(result["citation_pass"])

    def test_wrong_page(self):
        doc = Document(page_content=self.quote, metadata={"source": "기준.pdf", "page": 9})
        result = score_case(self.case, [doc], self.response)
        self.assertFalse(result["retrieval_pass"])
        self.assertFalse(result["citation_pass"])

    def test_missing_condition_term(self):
        result = score_case(self.case, [self.doc], {**self.response, "answer": "운임, 식비, 일비"})
        self.assertFalse(result["answer_terms_pass"])

    def test_forbidden_claim(self):
        result = score_case(self.case, [self.doc],
                            {**self.response, "answer": self.response["answer"] + " 무조건 지급"})
        self.assertFalse(result["answer_terms_pass"])

    def test_fabricated_quote(self):
        response = {**self.response, "sources": [{**self.doc.metadata, "quote": "없는 내용을 만들어낸 인용문입니다."}]}
        self.assertFalse(score_case(self.case, [self.doc], response)["citation_pass"])

    def test_refusal_is_not_correct_for_answerable_question(self):
        row = score_case(self.case, [self.doc], {"answer": UNKNOWN, "sources": []})
        self.assertFalse(row["automatic_pass"])

    def test_unknown_question(self):
        case = EvaluationCase(id="unknown", question="외계인 이름은?", answerable=False,
                              reference_answer=UNKNOWN, required_terms=[], expected_evidence=[])
        self.assertTrue(score_case(case, [], {"answer": UNKNOWN, "sources": []})["refusal_pass"])
        self.assertFalse(score_case(case, [], {"answer": "외계인은 갑입니다.", "sources": []})["refusal_pass"])

    def test_invalid_gold_stops_evaluation(self):
        check_gold([self.case], [self.doc])
        with self.assertRaises(ValueError):
            check_gold([self.case], [])

    def test_rates_use_applicable_cases_only(self):
        row = score_case(self.case, [self.doc], self.response)
        summary = summarize([row])
        self.assertEqual(summary["retrieval_pass"]["rate"], 1.0)
        self.assertIsNone(summary["refusal_pass"]["rate"])

    def test_app_rejects_invalid_evidence(self):
        answer = GroundedAnswer(answer="답변", supported=True,
                                evidence=[Evidence(source_id=1, quote="실제로 없는 원문 인용입니다.")])
        self.assertEqual(validate_answer(answer, [self.doc])["answer"], UNKNOWN)
        answer.evidence = [Evidence(source_id=2, quote=self.quote)]
        self.assertEqual(validate_answer(answer, [self.doc])["answer"], UNKNOWN)
        answer.evidence = [Evidence(source_id=1, quote=self.quote)]
        self.assertEqual(validate_answer(answer, [self.doc])["sources"][0]["page"], 8)


if __name__ == "__main__":
    unittest.main()
