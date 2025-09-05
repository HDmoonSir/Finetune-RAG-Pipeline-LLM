import os
import json
import glob
import google.generativeai as genai
from tqdm import tqdm
import math
import typing as tp

# 1. Gemini API 클라이언트 초기화 (환경 변수에서 API 키 로드)
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    genai.configure(api_key=api_key)
except (ValueError, KeyError) as e:
    print(f"API 키 설정 오류: {e}")
    # exit() # Do not exit here, let the calling script handle it


def generate_qa_pairs(
    prompt_content: str, gemini_model_name: str
) -> tp.Optional[tp.List[tp.Dict[str, str]]]:
    """Gemini 모델을 사용하여 Q&A 쌍을 생성합니다."""
    model = genai.GenerativeModel(
        gemini_model_name,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        ),
    )
    try:
        response = model.generate_content(prompt_content)
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:-4].strip()
        json_data = json.loads(json_text)
        return json_data.get("qa_pairs", list())
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"-- Q&A 생성 오류: {e}")
        return None
    except Exception as e:
        print(f"-- 예기치 않은 오류 발생: {e}")
        return None


def generate_unsupervised_text(
    prompt_content: str, gemini_model_name: str
) -> tp.Optional[str]:
    """Gemini 모델을 사용하여 비지도 학습용 텍스트를 생성합니다."""
    model = genai.GenerativeModel(gemini_model_name)
    try:
        response = model.generate_content(prompt_content)
        return response.text
    except Exception as e:
        print(f"-- 텍스트 생성 오류: {e}")
        return None


def format_for_llama3(
    qa_pairs: tp.List[tp.Dict[str, str]],
) -> tp.List[tp.Dict[str, str]]:
    """Q&A 쌍을 Llama 3의 Instruction Fine-tuning 형식으로 변환합니다."""
    formatted_data = list()
    for pair in qa_pairs:
        if (
            "question" not in pair
            or "answer" not in pair
            or not pair["question"]
            or not pair["answer"]
        ):
            continue
        instruction = pair["question"]
        response = pair["answer"]
        formatted_string = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )
        formatted_data.append({"text": formatted_string})
    return formatted_data


def main(
    mode: str,
    gemini_preprocess_model: str,
    data_dir: str,
    text_page_batch_size: int,
    qa_prompt_template: str,
    unsupervised_prompt_template: str,
    qa_dataset_path: str,
    unsupervised_dataset_path: str,
    default_output_dir: str,
) -> None:
    print(f"🚀'{mode}' 모드로 데이터 생성을 시작합니다.")

    jsonl_paths: tp.List[str] = glob.glob(
        os.path.join(data_dir, "**/*.jsonl"), recursive=True
    ) + glob.glob(os.path.join(data_dir, "*.jsonl"))
    jsonl_paths = sorted(list(set(jsonl_paths)))

    if not jsonl_paths:
        print("처리할 JSONL 파일이 없습니다.")
        return

    all_pages_by_pdf: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = {}
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data: tp.Dict[str, tp.Any] = json.loads(line)
                    source_pdf: tp.Optional[str] = data.get("source_pdf")
                    if source_pdf:
                        if source_pdf not in all_pages_by_pdf:
                            all_pages_by_pdf[source_pdf] = list()
                        all_pages_by_pdf[source_pdf].append(data)
                except json.JSONDecodeError:
                    print(
                        f"⚠️ Warning: {path} 파일의 일부 라인에서 JSON 디코딩 오류가 발생했습니다."
                    )

    for pdf in all_pages_by_pdf:
        all_pages_by_pdf[pdf].sort(key=lambda x: x.get("page_number", 0))

    all_generated_data: tp.List[tp.Any] = list()
    total_api_calls: int = 0

    for pdf_name, pages in all_pages_by_pdf.items():
        print(f"\n📄 {pdf_name} 문서 처리 중...")
        page_texts: tp.List[str] = [
            p["text"] for p in pages if p.get("text") and len(p["text"].strip()) > 50
        ]
        if not page_texts:
            continue

        num_batches: int = math.ceil(len(page_texts) / text_page_batch_size)
        for i in tqdm(
            range(num_batches), desc=f"{os.path.basename(pdf_name)} 처리 진행률"
        ):
            batch_texts: tp.List[str] = page_texts[
                i * text_page_batch_size : (i + 1) * text_page_batch_size
            ]
            combined_text: str = "\n\n---\n\n".join(batch_texts)

            if mode == "qa":
                prompt: str = qa_prompt_template.format(text=combined_text)
                result: tp.Optional[tp.List[tp.Dict[str, str]]] = generate_qa_pairs(
                    prompt, gemini_preprocess_model
                )
                if result:
                    all_generated_data.extend(result)
            else:  # unsupervised
                prompt = unsupervised_prompt_template.format(text=combined_text)
                result: tp.Optional[str] = generate_unsupervised_text(
                    prompt, gemini_preprocess_model
                )
                if result:
                    all_generated_data.append({"text": result})

            total_api_calls += 1

    print(f"\nTotal API calls made: {total_api_calls}")
    print(f"Total items generated before processing: {len(all_generated_data)}")

    if mode == "qa":
        unique_qa_pairs: tp.List[tp.Dict[str, str]] = list()
        processed_questions: tp.Set[str] = set()
        for pair in all_generated_data:
            question: tp.Optional[str] = pair.get("question")
            if question and question not in processed_questions:
                unique_qa_pairs.append(pair)
                processed_questions.add(question)
        final_data: tp.List[tp.Dict[str, str]] = format_for_llama3(unique_qa_pairs)
        output_file: str = qa_dataset_path
        print(f"Total unique Q&A pairs after deduplication: {len(unique_qa_pairs)}")
    else:
        final_data = all_generated_data
        output_file = unsupervised_dataset_path

    os.makedirs(default_output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ All data has been saved to '{output_file}'.")
    print(f"Total final records created: {len(final_data)}")
