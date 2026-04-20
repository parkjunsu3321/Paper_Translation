import os
import base64
import re
from openai import OpenAI
from tqdm import tqdm
from langchain_core.documents import Document
from config import Config

config = Config()
client = OpenAI(api_key=config.OPENAI_API_KEY)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path: str, image_type: str = "image") -> str:
    """Vision LLM으로 이미지 설명 생성"""
    base64_image = encode_image(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    if image_type == "table":
        prompt = "이 이미지의 내용을 자세히 설명해주세요."
    else:
        prompt = "이 이미지의 내용을 자세히 설명해주세요."

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
            ],
        }],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def extract_table_info(image_path: str) -> tuple:
    """
    Vision LLM으로 표 이미지에서 PDF 원문의 표 번호와 내용을 추출.

    Returns:
        (pdf_table_number, description)
        pdf_table_number: PDF에 표시된 표 번호 (int), 없으면 None
        description: 표 내용 설명 (str)
    """
    base64_image = encode_image(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    prompt = """이 표 이미지를 분석하세요.

표에 캡션이나 번호(예: "Table 1", "표 1", "TABLE 2" 등)가 보이면 첫 줄에 다음 형식으로 출력하세요:
TABLE_NUMBER: <숫자>

번호가 보이지 않으면 첫 줄을 생략하세요.

그 다음 표의 모든 행과 열 데이터를 빠짐없이 설명하세요."""

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_image}"}},
            ],
        }],
        max_tokens=1024,
    )
    text = response.choices[0].message.content

    # 첫 줄에서 PDF 원문 표 번호 파싱
    lines = text.splitlines()
    pdf_table_number = None
    if lines and lines[0].startswith("TABLE_NUMBER:"):
        try:
            pdf_table_number = int(lines[0].split(":", 1)[1].strip())
            text = "\n".join(lines[1:]).strip()
        except (ValueError, IndexError):
            pass

    return pdf_table_number, text


def get_table_documents(table_dir: str) -> tuple:
    """
    표 이미지를 별도 LangChain Document로 생성하고, 페이지 본문에 삽입할 참조 텍스트를 반환.
    표 번호는 PDF 원문에서 읽은 값을 사용하고, 인식 실패 시 순번으로 대체.

    Returns:
        page_references: {page_num: ["[표 N] ...", ...]}
        table_docs: [Document(...), ...]
    """
    page_pattern = re.compile(r"page_(\d+)_")
    page_references = {}
    table_docs = []
    fallback_number = 0  # PDF 번호 인식 실패 시 사용할 순번

    if not os.path.exists(table_dir):
        return page_references, table_docs

    # 파일명의 숫자를 기준으로 정렬 (알파벳 정렬 방지)
    def numeric_sort_key(fname):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', fname)]

    files = [f for f in sorted(os.listdir(table_dir), key=numeric_sort_key)
             if page_pattern.match(f)]

    for fname in tqdm(files, desc="Generating table descriptions"):
        page_num = int(page_pattern.match(fname).group(1))
        fallback_number += 1
        filepath = os.path.join(table_dir, fname)

        try:
            pdf_num, desc = extract_table_info(filepath)

            # PDF 원문 번호 우선, 없으면 순번 사용
            table_number = pdf_num if pdf_num is not None else fallback_number
            if pdf_num is None:
                print(f"  Info: {fname} - PDF 표 번호 미인식, 순번 {fallback_number} 사용")
            else:
                print(f"  Info: {fname} - PDF 표 번호 인식: Table {pdf_num}")

            # 표 내용을 독립 Document로 생성
            table_docs.append(Document(
                page_content=f"[표 {table_number} / 테이블 {table_number} / Table {table_number}]\n{desc}",
                metadata={
                    "type": "table",
                    "table_number": table_number,
                    "page": page_num,
                    "source": filepath,
                }
            ))

            # 페이지 본문에 삽입할 짧은 참조 텍스트
            ref = (
                f"[표 {table_number}] 이 위치에 표 {table_number}이 있습니다. "
                f"표 {table_number}의 세부 내용은 표 문서를 참조하세요. "
                f"(Table {table_number})"
            )
            page_references.setdefault(page_num, []).append(ref)

        except Exception as e:
            print(f"  Warning: {fname} 표 설명 생성 실패 - {e}")

    return page_references, table_docs


def get_image_descriptions(img_dir: str) -> dict:
    """이미지 설명 딕셔너리 반환. {page_num: ["[이미지 설명] ...", ...]}"""
    page_pattern = re.compile(r"page_(\d+)_")
    descriptions = {}

    if not os.path.exists(img_dir):
        return descriptions

    tasks = [
        (os.path.join(img_dir, f), int(page_pattern.match(f).group(1)))
        for f in sorted(os.listdir(img_dir))
        if page_pattern.match(f)
    ]

    for filepath, page_num in tqdm(tasks, desc="Generating image descriptions"):
        try:
            desc = describe_image(filepath, image_type="image")
            descriptions.setdefault(page_num, []).append(f"[이미지 설명] {desc}")
        except Exception as e:
            print(f"  Warning: {os.path.basename(filepath)} 설명 생성 실패 - {e}")

    return descriptions
