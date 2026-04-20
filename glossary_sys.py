import json
import os
import re
from typing import Dict, List

from openai import OpenAI
from ddg import Duckduckgo  # pip install duckduckgo-search-api

# -----------------------------
# 1. LLM 설정
# -----------------------------
_api_key = "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"
if not _api_key:
    raise RuntimeError("환경 변수 OPENAI_API_KEY 를 설정하세요.")

client = OpenAI(api_key=_api_key)


def call_llm(prompt: str, *, json_object: bool = False) -> str:
    kwargs = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": [
            {"role": "system", "content": "You are an expert in NLP terminology."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    res = client.chat.completions.create(**kwargs)
    content = res.choices[0].message.content
    return (content or "").strip()


def _parse_terms_from_llm_text(raw: str) -> List[str]:
    """마크다운·설명만 온 경우 보조 파싱."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*?\]", s)
        if not m:
            return []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    if isinstance(data, dict):
        for key in ("terms", "technical_terms", "term_list"):
            v = data.get(key)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
    return []


# -----------------------------
# 2. 핵심 용어 추출
# -----------------------------
def extract_terms(text: str) -> List[str]:
    prompt = f"""다음 영문에서 ML/NLP 논문에 쓸 만한 핵심 기술 용어(구)만 뽑아라.

규칙:
- 원문에 나온 표현을 우선 사용한다.
- JSON 한 개만 출력한다. 다른 설명·마크다운 금지.

스키마:
{{"terms": ["용어1", "용어2", ...]}}

본문:
{text}
"""
    try:
        raw = call_llm(prompt, json_object=True)
        data = json.loads(raw)
        terms = data.get("terms", [])
        if isinstance(terms, list) and terms:
            return [str(t).strip() for t in terms if str(t).strip()]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # 폴백: JSON 모드 실패 시 일반 호출 + 느슨한 파싱
    prompt2 = f"""List ML/NLP technical terms from this text as JSON array only.
Example: ["Recurrent neural networks", "convolutional neural networks"]

Text:
{text}
"""
    raw2 = call_llm(prompt2, json_object=False)
    return _parse_terms_from_llm_text(raw2)


# -----------------------------
# 3. 웹 검색으로 번역 후보 수집
# -----------------------------
ddg = Duckduckgo()


def search_translation(term: str) -> List[str]:
    query = f"{term} 한국어 번역"

    results = ddg.search(query)

    snippets = []
    for r in results.get("data", [])[:5]:
        snippets.append(r["description"])

    return snippets


# -----------------------------
# 4. LLM으로 번역 결정
# -----------------------------
def decide_translation(term: str, snippets: List[str]) -> str:
    context = "\n".join(snippets)

    prompt = f"""
Given the term and web search snippets, decide the most appropriate Korean translation.

Term:
{term}

Web Evidence:
{context}

Rules:
- Use standard academic translation
- Prefer commonly used terms in ML papers
- Return ONLY the Korean translation

Answer:
"""
    return call_llm(prompt)


# -----------------------------
# 5. Glossary 생성
# -----------------------------
def build_glossary(text: str) -> Dict[str, str]:
    glossary = {}

    terms = extract_terms(text)
    if not terms:
        print("(용어 추출 결과가 비었습니다. OPENAI_API_KEY·모델명·네트워크를 확인하세요.)")

    for term in terms:
        print(f"\n🔍 처리 중: {term}")

        snippets = search_translation(term)
        translation = decide_translation(term, snippets)

        glossary[term] = translation

    return glossary


# -----------------------------
# 6. 적용
# -----------------------------
def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    for k, v in glossary.items():
        pattern = re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE)
        text = pattern.sub(v, text)
    return text


# -----------------------------
# 7. 실행 예제
# -----------------------------
if __name__ == "__main__":
    text = "Recurrent neural networks, convolutional neural networks, Transformer, BERT, Attention, Sequence model"

    glossary = build_glossary(text)

    print("\n📘 Glossary:")
    print(glossary)

    result = apply_glossary(text, glossary)

    print("\n📝 적용 결과:")
    print(result)
