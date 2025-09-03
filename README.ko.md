# LLM 파인튜닝 및 RAG 파이프라인

이 프로젝트는 PDF 파일로부터 커스텀 데이터셋을 구축하고, LLM(거대 언어 모델)을 파인튜닝하며, RAG(검색 증강 생성) 파이프라인을 실행하기 위한 파이썬 스크립트 모음을 제공합니다.

## 스크립트 및 워크플로우

전체 프로세스는 다음과 같은 단계로 나뉩니다.

### 1. 데이터 준비

#### `extract_pdf_text.py`

PDF 파일에서 텍스트를 추출하여 JSONL 형식으로 저장합니다. 출력 파일의 각 라인은 PDF의 한 페이지에 해당합니다.

**사용법:**
```bash
python extract_pdf_text.py --file_path /path/to/your/document.pdf --start_page 1 --end_page 50
```
- `--file_path`: PDF 파일 경로.
- `--start_page` (선택 사항): 추출을 시작할 페이지 (기본값: 1).
- `--end_page` (선택 사항): 추출을 마칠 페이지 (기본값: 문서의 끝).
- `--output_file` (선택 사항): 출력할 `.jsonl` 파일의 이름.

#### `preprocess_with_gemini.py`

추출된 텍스트를 사용하여 Gemini API로 학습 데이터셋을 생성합니다. 두 가지 모드를 지원합니다:
- `qa`: Instruction 파인튜닝에 적합한 질문-답변 쌍을 생성합니다 (Llama 3 형식).
- `unsupervised`: 비지도 파인튜닝을 위한 일관성 있는 요약 텍스트를 생성합니다.

**사전 준비:**
Gemini API 키를 환경 변수로 설정해야 합니다:
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

**사용법:**
```bash
# 질문-답변 데이터셋 생성
python preprocess_with_gemini.py --mode qa

# 비지도 데이터셋 생성
python preprocess_with_gemini.py --mode unsupervised
```
이 스크립트는 `data/` 디렉토리에 있는 모든 `.jsonl` 파일을 처리하여 `gemini_generated_qa_dataset.jsonl` 또는 `gemini_generated_unsupervised_dataset.jsonl` 파일을 생성합니다.

### 2. 모델 파인튜닝

`MLP-KTLim/llama-3-Korean-Bllossom-8B` 모델을 파인튜닝하기 위한 두 가지 스크립트를 제공합니다.

#### `train_llm.py` (표준 방식)

LoRA를 사용하여 모델을 파인튜닝합니다.

**사용법:**
```bash
python train_llm.py --dataset_path /path/to/your_dataset.jsonl
```
- `--dataset_path`: 생성된 `.jsonl` 데이터셋 파일의 경로.

#### `train_unsloth.py` (최적화 방식)

Unsloth 라이브러리를 사용하여 메모리를 절약하고 더 빠른 속도로 학습을 진행하는 최적화 버전입니다.

**사용법:**
```bash
python train_unsloth.py --dataset_path /path/to/your_dataset.jsonl
```
- `--dataset_path`: 생성된 `.jsonl` 데이터셋 파일의 경로.

### 3. RAG 추론

#### `rag_pipeline.py`

KorQuAD 데이터셋을 지식 베이스로 사용하여 RAG 파이프라인을 실행하는 통합 스크립트입니다. 다양한 모델 로딩 전략을 지원합니다.

**사용법:**
```bash
python rag_pipeline.py {model_type} "{your_question}"
```

**위치 인자 (Positional Arguments):**

- `model_type`: 추론에 사용할 모델의 타입. 다음 중에서 선택합니다:
  - `api`: Gemini API (`gemini-1.5-flash-latest`)를 사용합니다. `GEMINI_API_KEY` 환경 변수 설정이 필요합니다.
  - `local`: 전체 정밀도(full-precision) 로컬 모델 (`MLP-KTLim/llama-3-Korean-Bllossom-8B`)을 사용합니다. Unsloth 호환성 문제가 있는 Volta 아키텍처의 GPU(예: Titan V)에 권장됩니다.
  - `local-quantized`: Unsloth를 통해 4비트 양자화된 로컬 모델을 사용하여 더 빠른 속도와 낮은 메모리 사용량을 제공합니다. 호환되는 NVIDIA GPU(Ampere, Turing, Ada 등)에 권장됩니다.

- `your_question`: RAG 시스템에 질문할 내용. 반드시 큰따옴표("")로 감싸야 합니다.

- `--lora-path`: LoRA 어댑터의 경로 (선택 사항). 로컬 모델에만 적용됩니다.

**실행 예시:**

1.  **API 모델 테스트:**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    python rag_pipeline.py api "파우스트의 작가는 누구인가?"
    ```

2.  **로컬 모델 테스트 (비양자화, Volta/호환성용):**
    ```bash
    python rag_pipeline.py local "임진왜란이 발발한 연도는 언제야?"
    ```

3.  **로컬 모델 테스트 (양자화, 호환 GPU용):**
    ```bash
    python rag_pipeline.py local-quantized "세종대왕이 한글을 창제한 연도는?"
    ```

#### `rag_interactive_cli.py`

RAG 파이프라인을 위한 대화형(Interactive) 명령줄 인터페이스입니다. 이 스크립트는 시작 시 모델과 지식 베이스를 한 번만 로드하여, 재로딩 없이 연속적인 질문을 가능하게 합니다.

**사용법:**
```bash
python rag_interactive_cli.py {model_type}
```

**인자:**
- `model_type`: 사용할 모델의 타입. `rag_pipeline.py`와 동일한 선택지: `api`, `local`, `local-quantized`.

- `--lora-path`: LoRA 어댑터의 경로 (선택 사항). 로컬 모델에만 적용됩니다.

**상호작용:**
스크립트 시작 후, 질문 프롬프트가 나타납니다. 질문을 입력하고 Enter를 누르세요. 종료하려면 `exit` 또는 `quit`를 입력하고 Enter를 누르세요.

**실행 예시:**

1.  **API 모델 (대화형):**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    python rag_interactive_cli.py api
    ```
    (프롬프트에 질문 입력)

2.  **로컬 모델 (비양자화, 대화형):**
    ```bash
    python rag_interactive_cli.py local
    ```
    (프롬프트에 질문 입력)

## 프로젝트 설정

1.  **저장소 복제(Clone):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```
3.  각 스크립트의 사용법에 따라 실행합니다.

## 라이선스

이 프로젝트는 Llama 3 커뮤니티 라이선스에 따라 제공되는 Llama 3를 활용합니다. 전체 라이선스 내용은 [LICENSE](LICENSE) 파일을 참고해 주세요.
