import os
import json
import glob
import google.generativeai as genai
from tqdm import tqdm
import math
import argparse
import typing as tp

# 1. Gemini API 클라이언트 초기화
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    genai.configure(api_key=api_key)
except (ValueError, KeyError) as e:
    print(f"API 키 설정 오류: {e}")
    exit()

def generate_qa_pairs(prompt_content: str) -> tp.Optional[tp.List[tp.Dict[str, str]]]:
    """Gemini 모델을 사용하여 Q&A 쌍을 생성합니다."""
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
    )
    try:
        response = model.generate_content(prompt_content)
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:-4].strip()
        json_data = json.loads(json_text)
        return json_data.get('qa_pairs', [])
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"-- Q&A 생성 오류: {e}")
        return None
    except Exception as e:
        print(f"-- 예기치 않은 오류 발생: {e}")
        return None

def generate_unsupervised_text(prompt_content: str) -> tp.Optional[str]:
    """Gemini 모델을 사용하여 비지도 학습용 텍스트를 생성합니다."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt_content)
        return response.text
    except Exception as e:
        print(f"-- 텍스트 생성 오류: {e}")
        return None

def format_for_llama3(qa_pairs: tp.List[tp.Dict[str, str]]) -> tp.List[tp.Dict[str, str]]:
    """Q&A 쌍을 Llama 3의 Instruction Fine-tuning 형식으로 변환합니다."""
    formatted_data = []
    for pair in qa_pairs:
        if 'question' not in pair or 'answer' not in pair or not pair['question'] or not pair['answer']:
            continue
        instruction = pair['question']
        response = pair['answer']
        formatted_string = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>")
        formatted_data.append({"text": formatted_string})
    return formatted_data

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini를 사용하여 텍스트 데이터셋을 생성합니다.")
    parser.add_argument(
        '--mode',
        type=str,
        default='qa',
        choices=['qa', 'unsupervised'],
        help='생성 모드를 선택합니다. `qa`는 질의응답 쌍을, `unsupervised`는 비지도학습용 텍스트를 생성합니다.'
    )
    args = parser.parse_args()

    print(f"🚀'{args.mode}' 모드로 데이터 생성을 시작합니다.")

    jsonl_paths: tp.List[str] = glob.glob("data/**/*.jsonl", recursive=True) + glob.glob("data/*.jsonl")
    jsonl_paths = sorted(list(set(jsonl_paths)))

    if not jsonl_paths:
        print("처리할 JSONL 파일이 없습니다.")
        return

    all_pages_by_pdf: tp.Dict[str, tp.List[tp.Dict[str, tp.Any]]] = {}
    for path in jsonl_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data: tp.Dict[str, tp.Any] = json.loads(line)
                    source_pdf: tp.Optional[str] = data.get('source_pdf')
                    if source_pdf:
                        if source_pdf not in all_pages_by_pdf:
                            all_pages_by_pdf[source_pdf] = []
                        all_pages_by_pdf[source_pdf].append(data)
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: {path} 파일의 일부 라인에서 JSON 디코딩 오류가 발생했습니다.")

    for pdf in all_pages_by_pdf:
        all_pages_by_pdf[pdf].sort(key=lambda x: x.get('page_number', 0))

    all_generated_data: tp.List[tp.Any] = []
    total_api_calls: int = 0
    TEXT_PAGE_BATCH_SIZE: int = 15

    qa_prompt_template: str = """아래 [문서 내용]은 '---'로 구분된 여러 텍스트 섹션을 포함하고 있습니다. 이 내용을 바탕으로, 다양하고 핵심적인 Q&A 쌍 25개를 \"qa_pairs\"를 키로 하는 JSON 배열 형식으로 생성해 주세요. 질문과 답변은 모두 한국어로 작성해야 합니다. 출력은 오직 JSON 배열이어야 합니다.

[문서 내용]
{text}"""
    unsupervised_prompt_template: str = """아래 [문서 내용]은 '---'로 구분된 여러 텍스트 섹션을 포함하고 있습니다. 이 내용을 바탕으로, 문서의 핵심 주제와 정보를 포괄하는 상세하고 구조화된 요약 텍스트를 생성해 주세요. 원본의 전문적인 스타일과 톤을 유지하며, 여러 단락으로 구성된 가독성 높은 글을 작성해야 합니다. 출력은 오직 생성된 텍스트여야 합니다.

[문서 내용]
{text}"""

    for pdf_name, pages in all_pages_by_pdf.items():
        print(f"\n📄 {pdf_name} 문서 처리 중...")
        page_texts: tp.List[str] = [p['text'] for p in pages if p.get('text') and len(p['text'].strip()) > 50]
        if not page_texts:
            continue

        num_batches: int = math.ceil(len(page_texts) / TEXT_PAGE_BATCH_SIZE)
        for i in tqdm(range(num_batches), desc=f"{os.path.basename(pdf_name)} 처리 진행률"):
            batch_texts: tp.List[str] = page_texts[i * TEXT_PAGE_BATCH_SIZE : (i + 1) * TEXT_PAGE_BATCH_SIZE]
            combined_text: str = "\n\n---\n\n".join(batch_texts)

            if args.mode == 'qa':
                prompt: str = qa_prompt_template.format(text=combined_text)
                result: tp.Optional[tp.List[tp.Dict[str, str]]] = generate_qa_pairs(prompt)
                if result:
                    all_generated_data.extend(result)
            else: # unsupervised
                prompt = unsupervised_prompt_template.format(text=combined_text)
                result: tp.Optional[str] = generate_unsupervised_text(prompt)
                if result:
                    all_generated_data.append({"text": result})
            
            total_api_calls += 1

    print(f"\nTotal API calls made: {total_api_calls}")
    print(f"Total items generated before processing: {len(all_generated_data)}")

    if args.mode == 'qa':
        unique_qa_pairs: tp.List[tp.Dict[str, str]] = []
        processed_questions: tp.Set[str] = set()
        for pair in all_generated_data:
            question: tp.Optional[str] = pair.get('question')
            if question and question not in processed_questions:
                unique_qa_pairs.append(pair)
                processed_questions.add(question)
        final_data: tp.List[tp.Dict[str, str]] = format_for_llama3(unique_qa_pairs)
        output_file: str = 'gemini_generated_qa_dataset.jsonl'
        print(f"Total unique Q&A pairs after deduplication: {len(unique_qa_pairs)}")
    else:
        final_data = all_generated_data
        output_file = 'gemini_generated_unsupervised_dataset.jsonl'

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ All data has been saved to '{output_file}'.")
    print(f"Total final records created: {len(final_data)}")

if __name__ == "__main__":
    main()