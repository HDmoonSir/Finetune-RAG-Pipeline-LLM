# LLM 파인튜닝 및 RAG 파이프라인

이 프로젝트는 PDF 파일로부터 커스텀 데이터셋을 구축하고, LLM(거대 언어 모델)을 파인튜닝하며, RAG(검색 증강 생성) 파이프라인을 실행하기 위한 파이썬 스크립트 모음을 제공합니다.

## 프로젝트 구조

이 프로젝트는 다음과 같은 디렉토리 구조로 구성되어 있습니다:

- `src/`: 모든 파이썬 소스 코드를 포함합니다.
  - `data_processing/`: 데이터 추출 및 전처리 스크립트.
  - `training/`: 모델 파인튜닝 및 LoRA 병합 스크립트.
  - `rag/`: RAG 파이프라인, 벡터 저장소 구축 및 추론 스크립트.
  - `evaluation/`: RAG 파이프라인 평가 스크립트.
- `run_data.py`: 데이터 처리 관련 작업을 위한 실행 스크립트.
- `run_training.py`: 모델 학습 및 병합 관련 작업을 위한 실행 스크립트.
- `run_rag.py`: RAG 파이프라인 관련 작업을 위한 실행 스크립트.
- `run_evaluation.py`: 평가 관련 작업을 위한 실행 스크립트.
- `data/`: PDF, 원본 JSONL 파일 등 원시 데이터를 위한 디렉토리.
- `data_result/`: 생성된 데이터셋, 벡터 저장소, 모델, 평가 결과 등을 위한 디렉토리.
- `config.py`: 프로젝트의 메인 설정 파일.

## 워크플로우 및 사용법

전체 프로세스는 다음과 같은 단계로 나뉩니다. 모든 스크립트는 프로젝트의 루트 디렉토리에서 실행해야 합니다.

### 1. 데이터 준비

#### 가. PDF에서 텍스트 추출
- **스크립트:** `run_data.py extract-pdf`
- **설명:** PDF 파일에서 텍스트를 추출하여 JSONL 형식으로 저장합니다.
- **사용법:**
  ```bash
  python3.11 run_data.py extract-pdf --file_path /path/to/your/document.pdf
  ```
- **인자:**
  - `--file_path`: PDF 파일 경로.
  - `--start_page` (선택 사항): 추출을 시작할 페이지 (기본값: 1).
  - `--end_page` (선택 사항): 추출을 마칠 페이지 (기본값: 문서의 끝).

#### 나. 학습 데이터 생성
- **스크립트:** `run_data.py preprocess-gemini`
- **설명:** 추출된 텍스트를 사용하여 Gemini API로 학습 데이터셋을 생성합니다.
- **사전 준비:** `export GEMINI_API_KEY="YOUR_API_KEY"`
- **사용법:**
  ```bash
  # 질문-답변(SFT) 데이터셋 생성
  python3.11 run_data.py preprocess-gemini --mode qa

  # 비지도 데이터셋 생성
  python3.11 run_data.py preprocess-gemini --mode unsupervised
  ```

### 2. 모델 파인튜닝

#### 가. 모델 파인튜닝
- **스크립트:** `run_training.py train` (표준 방식) 또는 `run_training.py train-unsloth` (최적화 방식)
- **설명:** LoRA를 사용하여 모델을 파인튜닝합니다. 지도 학습(SFT)과 비지도 학습 모드를 모두 지원합니다.
- **사용법:**
  ```bash
  # 지도 파인튜닝 (SFT)
  python3.11 run_training.py train-unsloth --dataset_path <QA_데이터셋_경로> --mode sft

  # 비지도 지속 사전학습
  python3.11 run_training.py train-unsloth --dataset_path <비지도_데이터셋_경로> --mode unsupervised
  ```

#### 나. LoRA 어댑터 병합
- **스크립트:** `run_training.py merge`
- **설명:** 학습된 LoRA 어댑터를 베이스 모델에 병합하여 독립적인 새 모델을 생성합니다. 비지도 학습 후 도메인 적응 모델을 만들 때 유용합니다.
- **사용법:**
  ```bash
  python3.11 run_training.py merge --lora_path <LoRA_어댑터_경로> --output_dir <병합된_모델_저장_경로>
  ```

### 3. RAG 파이프라인

#### 가. 벡터 저장소 구축
- **스크립트:** `run_rag.py build-db`
- **설명:** RAG 파이프라인을 위해 원본 문서들로부터 FAISS 벡터 저장소를 구축합니다.
- **사용법:**
  ```bash
  python3.11 run_rag.py build-db --input_dir data/
  ```

#### 나. RAG 추론 실행
- **스크립트:** `run_rag.py pipeline` (단일 질문) 또는 `run_rag.py cli` (대화형 채팅)
- **설명:** 지정된 모델과 지식 베이스를 사용하여 RAG 파이프라인을 실행합니다.
- **사용법 (대화형 CLI):**
  ```bash
  # API 모델과 커스텀 벡터 저장소 사용
  export GEMINI_API_KEY="YOUR_API_KEY"
  python3.11 run_rag.py cli --model_type api --knowledge_base data_result/vector_store

  # 로컬 모델과 커스텀 벡터 저장소 사용
  python3.11 run_rag.py cli --model_type local --knowledge_base data_result/vector_store
  ```
- **주요 인자:**
  - `--model_type`: `api`, `local`, `local-quantized`.
  - `--lora-path`: LoRA 어댑터의 경로 (선택 사항).
  - `--knowledge_base`: 사용할 FAISS 벡터 저장소 경로, 또는 KorQuAD를 사용하려면 `default`.

### 4. 평가

- **스크립트:** `run_evaluation.py`
- **설명:** QA 데이터셋으로 RAG 파이프라인을 평가하고 ROUGE/BLEU 점수를 계산합니다.
- **사용법:**
  ```bash
  python3.11 run_evaluation.py --model_type api --knowledge_base data_result/vector_store --num_samples 20
  ```

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
3.  위 워크플로우 단계에 따라 데이터를 준비하고, 모델을 학습시키고, 파이프라인을 실행합니다.

## 라이선스

이 프로젝트는 Llama 3 커뮤니티 라이선스에 따라 제공되는 Llama 3를 활용합니다. 전체 라이선스 내용은 [LICENSE](LICENSE) 파일을 참고해 주세요.
