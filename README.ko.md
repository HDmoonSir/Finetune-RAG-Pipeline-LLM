---

# LLM 파인튜닝 & RAG 파이프라인

이 프로젝트는 PDF 파일로부터 커스텀 데이터셋을 구축하고, 이를 활용해 대형 언어 모델(LLM)을 파인튜닝하며, RAG(Retrieval-Augmented Generation) 파이프라인을 실행하는 일련의 Python 스크립트를 제공합니다.

---

## 📁 프로젝트 구조

프로젝트는 다음과 같이 구성되어 있습니다:

* `src/`: 주요 Python 소스 코드 디렉토리

  * `data_preprocessing/`: 데이터 추출 및 전처리 관련 스크립트
  * `training/`: 모델 파인튜닝 및 LoRA 병합 관련 스크립트
  * `rag/`: RAG 파이프라인 관련 스크립트
  * `evaluation/`: RAG 결과 평가용 스크립트
  * `inference/`: 모델 추론 및 실험 로딩 관련 스크립트
  * `utils/`: 유틸리티 스크립트 및 설정 로더
* `run_data.py`: 데이터 처리 작업 실행 스크립트
* `run_training.py`: 모델 학습 및 병합 작업 실행 스크립트
* `run_rag.py`: RAG 파이프라인 실행 스크립트
* `run_evaluation.py`: 평가 작업 실행 스크립트
* `data/`: 원본 데이터 (PDF, JSONL 등)
* `data_result/`: 결과 데이터 저장 디렉토리 (전처리 데이터, 벡터스토어, 모델, 평가 결과 등)
* `config/`: 데이터 처리, 학습, 추론, 평가를 위한 YAML 설정 파일 저장 디렉토리

---

## ⚙️ YAML 기반 설정 시스템

이 프로젝트는 작업 실행 시 **YAML 설정 파일을 기반으로 구성**되며, 많은 파라미터를 명령어에 직접 입력할 필요 없이 `--config` 옵션만으로 실행할 수 있습니다. 재현성과 설정 관리가 쉬워지는 장점이 있습니다.

### 사용 방법

1. **설정 파일 위치 확인:** `config/` 디렉토리 아래 작업 유형별로 정리되어 있습니다.
   (예: `config/data/`, `config/train/`, `config/eval/`, `config/inference/`)
2. **설정 파일 선택 및 수정:** 필요한 설정을 직접 수정하여 사용합니다.
3. **스크립트 실행:** 아래와 같이 `--config` 인자를 사용합니다.

**예시:**

```bash
python3.11 run_training.py --config config/train/llama-8b-sft.yaml
```

---

## 🧭 전체 작업 흐름

전체 파이프라인은 다음과 같은 순서로 구성되어 있습니다. 모든 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.

---

### 1. 데이터 준비

데이터 준비 작업은 `run_data.py`를 통해 하위 명령어(subcommand) 방식으로 수행됩니다.

#### a. PDF 텍스트 추출

* **스크립트:** `run_data.py`
* **하위 명령어:** `extract-pdf`
* **기능:** PDF 파일에서 텍스트를 추출하여 JSONL 형식으로 저장
* **실행 명령어:**

  ```bash
  python3.11 run_data.py extract-pdf --file_path data/your_file.pdf --output_file data/output.jsonl
  ```

#### b. 학습 데이터 생성

* **스크립트:** `run_data.py`
* **하위 명령어:** `preprocess-gemini`
* **기능:** Gemini API를 이용해 추출된 텍스트로 QA 또는 비지도 학습 데이터 생성
* **사전 설정:**

  ```bash
  export GEMINI_API_KEY="YOUR_API_KEY"
  ```
* **설정 파일:** `config/data/preprocess_gemini.yaml`
* **실행 명령어:**

  ```bash
  python3.11 run_data.py preprocess-gemini --config config/data/preprocess_gemini.yaml
  ```

#### c. 벡터스토어 구축

* **스크립트:** `run_data.py`
* **하위 명령어:** `build-db`
* **기능:** 입력 문서로부터 임베딩을 생성하고 FAISS 기반 벡터스토어 생성
* **설정 파일:** `config/data/build_vector_store.yaml`
* **실행 명령어:**

  ```bash
  python3.11 run_data.py build-db --config config/data/build_vector_store.yaml
  ```

---

### 2. 모델 파인튜닝

모델 학습은 두 단계로 진행됩니다: **도메인 적응 (Unsupervised)** → **태스크 학습 (Supervised Fine-Tuning)**

#### a. 1단계: 비지도 도메인 적응

* **목적:** 도메인 특화 텍스트로 LLM을 사전 적응시킵니다.
* **스크립트:** `run_training.py`
* **설정:** `training.mode`가 `unsupervised`인 YAML
* **출력:** 도메인 정보가 반영된 LoRA 어댑터 (`unsupervised_lora_adapter`)
* **예시 명령어:**

  ```bash
  python3.11 run_training.py --config config/train/llama-8b-unsupervised.yaml
  ```

#### b. 2단계: 감독 학습 (SFT)

* **목적:** 구체적인 작업(예: Q\&A)에 맞춰 모델을 지도 학습시킵니다.
* **스크립트:** `run_training.py`
* **설정:** `training.mode: sft`, `model.unsupervised_lora_path`를 통해 1단계 결과를 활용
* **출력:** 태스크 지식이 포함된 새로운 LoRA 어댑터 (`sft_lora_adapter`)
* **예시 명령어:**

  ```bash
  python3.11 run_training.py --config config/train/llama-8b-sft.yaml
  ```

---

### 3. RAG 파이프라인 및 평가

RAG 파이프라인과 평가에서도 LoRA 어댑터를 메모리 상에서 병합해 사용할 수 있습니다.

#### a. RAG 기반 추론

* **스크립트:** `run_rag.py`
* **하위 명령어:** `cli` (대화형 모드) 또는 `pipeline` (단일 질문)
* **설정:** `config/inference/local_rag.yaml`을 통해 모델 및 벡터스토어 설정
* **RAG 실행 예시 (CLI):**

  ```bash
  python3.11 run_rag.py cli --config config/inference/local_rag.yaml
  ```

* **RAG 실행 예시 (단일 질문):**

  ```bash
  python3.11 run_rag.py pipeline --config config/inference/local_rag.yaml --question "질문 내용"
  ```

#### b. 평가 실행

* **스크립트:** `run_evaluation.py`
* **기능:** `config/eval/default_eval.yaml`에 정의된 메트릭을 사용하여 RAG 성능 평가
* **평가 실행 예시:**

  ```bash
  python3.11 run_evaluation.py --config config/eval/default_eval.yaml
  ```

---

## ⚙️ 프로젝트 설치 방법

1. **저장소 클론**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **의존성 설치**

   ```bash
   pip install -r requirements.txt
   ```

3. 데이터 준비 → 모델 학습 → RAG 실행 순서로 워크플로우를 진행합니다.

---

## 📄 라이선스

본 프로젝트는 Meta의 **Llama 3** 모델을 기반으로 하며, [Llama 3 Community License](LICENSE)를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.

---
