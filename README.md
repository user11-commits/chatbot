# 공무원 여비 문서 RAG 챗봇

`DATA` 폴더의 문서를 검색하여 질문에 답하는 **Python 3.11 · Streamlit · OpenAI** 기반 챗봇입니다. 답변 아래에 출처 파일명, PDF 페이지 번호, 원문 근거를 표시합니다.

## 주요 기능

- DATA와 하위 폴더의 PDF, TXT, MD, CSV, JSON 읽기
- 문서 조각의 임베딩과 유사도 검색
- 검색된 문서를 바탕으로 한국어 답변 생성
- 원문 인용 검증: 인용이 검색 원문에 없으면 답변 거절
- 대화 기록 표시와 대화 지우기
- 정답표 기반 평가 및 JSON·CSV 보고서 생성
- Streamlit Cloud Secrets와 로컬 `.env` 지원

각 질문은 독립적으로 검색합니다. 이전 답변을 가리키는 표현 대신 질문에 필요한 내용을 모두 적어 주세요.

## 사용 기술

| 구성 | 기술 / 설정 |
|---|---|
| Python | 3.11 |
| 환경 및 의존성 관리 | uv, pyproject.toml, uv.lock |
| 화면 | Streamlit |
| 임베딩 | OpenAI `text-embedding-3-small` |
| 답변 모델 | OpenAI `gpt-4o-mini`, temperature=0 |
| 벡터 저장소 | LangChain `InMemoryVectorStore` |
| 문서 분할 | `RecursiveCharacterTextSplitter`: 최대 1,000자, 150자 겹침 |
| 검색 | 질문마다 관련 문서 조각 최대 6개 |
| PDF 추출 | pypdf, 페이지별 텍스트 추출 |

LangChain의 `invoke()`와 Runnable 파이프라인, 구조화된 출력을 사용합니다. LLMChain, ConversationChain, RetrievalQA는 사용하지 않습니다.

## 프로젝트 구성

```text
chatbot/
├── app.py                         # Streamlit 챗봇
├── evaluate.py                    # 실제 RAG 평가 도구
├── DATA/                          # 검색 대상 문서
├── evaluation/
│   ├── cases.json                 # 질문·정답·출처 기준
│   └── results/                   # 평가 보고서
├── scripts/check_environment.py   # 패키지 및 기본 기능 확인
├── tests/                         # 평가와 Secrets 테스트
├── .streamlit/secrets.toml.example # 공개용 빈 Secrets 예시
├── .env.example                   # 공개용 빈 .env 예시
├── .gitignore
├── .python-version
├── pyproject.toml
└── uv.lock
```

`.env`, `.streamlit/secrets.toml`, `.venv`는 Git에서 제외됩니다.

## 로컬 실행

### 1. 프로젝트 다운로드와 환경 준비

[uv](https://docs.astral.sh/uv/getting-started/installation/)와 Git을 준비하고 PowerShell에서 실행하세요.

```powershell
git clone https://github.com/user11-commits/chatbot.git
cd chatbot
uv sync --locked --no-editable
```

프로젝트는 `.python-version`에 Python 3.11을 지정합니다. Windows의 한글 경로와 Python 3.11에서 editable 설치 경로를 읽을 때 발생하는 인코딩 문제를 피하기 위해 `--no-editable`을 사용합니다.

### 2. API 키 설정

처음 설정할 때만 예제 파일을 복사하세요. 기존 `.env`가 있다면 내용을 유지하고 편집합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일의 빈 항목에 자신의 키를 입력합니다.

```dotenv
OPENAI_API_KEY=여기에_실제_API_키
```

키는 코드·커밋·README에 넣지 마세요. 실제 키가 있는 파일은 업로드하지 않습니다.

### 3. 앱 실행

```powershell
uv run --no-editable --locked streamlit run app.py
```

터미널에 표시되는 주소를 열고 **문서 준비**를 누르세요. 문서 임베딩이 끝나면 질문할 수 있습니다.

예시 질문:

- 국내 출장 여비에는 어떤 항목이 포함되나요?
- 근무지 내 출장의 여비 지급 기준은 무엇인가요?
- 자가용을 이용한 출장에서 운임은 어떻게 지급하나요?

문서 준비와 질문 처리에는 OpenAI API 호출 비용이 발생합니다. 벡터는 세션 메모리에 보관되므로 앱 재시작, 새 세션, 문서 또는 키 변경 후에는 다시 준비해야 합니다.

## Streamlit Community Cloud 배포

1. [Streamlit Community Cloud](https://share.streamlit.io)에 GitHub 계정으로 로그인합니다.
2. **Create app**에서 다음 항목을 입력합니다.
   - Repository: `user11-commits/chatbot`
   - Branch: `main`
   - Main file path: `app.py`
3. **Advanced settings**에서 Python **3.11**을 선택합니다.
4. **Secrets** 입력란에 다음 TOML 형식으로 키를 넣습니다.
5. **Save → Deploy**를 선택합니다.

```toml
OPENAI_API_KEY = "여기에_실제_API_키"
```

`[openai]` 같은 섹션 없이 최상위 항목으로 입력하세요. 앱은 `st.secrets["OPENAI_API_KEY"]`를 우선 읽고, 설정이 없으면 로컬 `.env`를 사용합니다. Cloud에서는 `.env`를 업로드하지 않습니다.

배포 후에는 앱의 **Settings → Secrets**에서 키를 변경할 수 있습니다. Cloud는 저장소의 `uv.lock`으로 의존성을 설치하므로 별도 `requirements.txt`는 추가하지 않습니다.

공식 문서: [배포 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [Secrets 설정](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), [의존성 설치](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## 답변 생성과 근거 검증

1. PDF를 페이지별로 읽고 파일명과 PDF 페이지 번호를 저장합니다.
2. 텍스트를 조각으로 나누고 임베딩합니다.
3. 질문을 같은 임베딩 모델로 변환하여 관련 조각을 검색합니다.
4. 시스템 프롬프트에 문서만 사용하기, 추측 금지, 원문 인용, 조건·예외 보존 규칙을 전달합니다.
5. 검색 문서와 질문을 모델에 전달해 답변과 근거를 구조화된 형태로 받습니다.
6. 자료 번호와 인용문이 원문에 있는지 검사한 뒤 답변과 출처를 표시합니다.

답변 근거가 없거나 인용 검증이 실패하면 다음 문구를 표시합니다.

> 제공된 문서에서 질문에 대한 근거를 찾을 수 없습니다.

원문에 인용문이 존재한다는 사실만으로 답변의 모든 주장이 정확하다고 보장되지는 않습니다. 중요한 판단에는 원문과 적용 조건을 함께 확인하세요.

## 검증 도구

### 설치 환경 확인

```powershell
uv run --no-editable --locked python -W error scripts/check_environment.py
```

실제 API 요청 없이 import, 문서 분할, 프롬프트 처리, dotenv 처리, OpenAI 클라이언트 생성을 확인합니다.

### 평가 로직 테스트

```powershell
uv run --no-editable --locked python -W error::DeprecationWarning -W error::FutureWarning -m unittest discover -s tests -v
```

실제 키와 API 호출 없이 검색·출처 판정, 답변 거절, Cloud Secrets 및 로컬 키 처리 로직을 검사합니다.

### 정답표 확인

`evaluation/cases.json`에 PDF를 직접 보고 정한 질문, 기준 답변, 필수 단어, 원문 근거를 기록합니다. API를 호출하기 전에 정답표의 파일명·페이지·인용이 실제 문서와 일치하는지 확인하세요.

```powershell
uv run --no-editable --locked python evaluate.py --check-only
```

### 실제 RAG 평가

```powershell
uv run --no-editable --locked python evaluate.py
```

전체 문서를 임베딩하고 질문별로 API를 호출합니다. 결과는 `evaluation/results/latest.json`과 `latest.csv`에 저장됩니다. 이전 결과를 보존하려면 파일명을 지정하세요.

```powershell
uv run --no-editable --locked python evaluate.py --output evaluation/results/baseline.json
```

| 지표 | 검사 내용 |
|---|---|
| retrieval_pass | 지정한 원문 근거가 검색 결과에 포함되는지 |
| answer_terms_pass | 필수 단어가 있고 금지 단어가 없는지 |
| citation_pass | 인용이 검색 원문에 있고 정답 출처·근거와 겹치는지 |
| refusal_pass | 문서 밖 질문에 답변을 거절하고 출처를 만들지 않는지 |

- `required_terms`의 각 그룹에서 최소 한 표현이 답변에 있어야 합니다. `[["운임", "교통비"], ["숙박비"]]`는 운임 또는 교통비와 숙박비가 모두 필요하다는 뜻입니다.
- `expected_evidence.page`는 문서에 인쇄된 쪽수가 아니라 **PDF 첫 페이지부터 1로 세는 페이지 번호**입니다.
- 근거를 여러 개 지정하면 모두 검색·인용해야 통과합니다.
- API 오류도 실패로 집계합니다. 해당 평가 유형이 없으면 비율은 `null`입니다.
- 종료 코드: 통과 `0`, 평가 실패 `1`, 정답표·환경 오류 `2`.

자동 판정은 단어와 원문 일치를 검사합니다. 의미·수치·조건·예외는 사람이 보고서의 답변과 검색 원문을 비교해야 합니다. CSV의 `manual_review`에 `pass` 또는 `fail`, `manual_notes`에 이유를 기록하세요.

현재 기본 평가 세트는 질문 두 건입니다. 저장된 예제 결과에서는 문서 밖 질문 거절은 통과했지만, 국내 출장 질문은 검색·출처 검사에 실패했습니다. 이 소규모 결과를 전체 정확도로 해석하지 말고 조건·금액·예외·사례 질문을 추가하여 비교하세요.

## 현재 한계와 문제 해결

- **표·문답 구조:** 페이지별 텍스트 추출과 길이 기준 분할을 사용하여 표, 질문·답변, 조건·예외가 떨어질 수 있습니다. 목차도 검색에 포함되므로 구조에 맞춘 전처리와 검색 개선이 필요합니다.
- **스캔 PDF:** 이미지에 있는 글자는 OCR 없이 읽지 못합니다. 텍스트 추출 불가 페이지는 화면에서 안내합니다.
- **문서의 시점:** 저장된 문서의 내용을 답하므로 현재 규정인지 별도로 확인해야 합니다.
- **후속 질문:** “그 답변의 출처는?” 대신 원래 질문을 다시 적으세요. 출처는 답변 아래에 자동 표시됩니다.
- **Cloud에서 예전 `.env` 안내가 표시됨:** Secrets 지원 코드가 GitHub `main`에 반영되었는지 확인하고 앱을 다시 시작하세요.
- **OpenAI 인증 또는 한도 오류:** Secrets 또는 `.env`의 키, OpenAI 계정의 잔액과 요청 한도를 확인하세요.
- **지원하지 않는 파일:** 지원 형식으로 변환하세요. 앱은 미지원 파일을 조용히 생략하지 않습니다.
