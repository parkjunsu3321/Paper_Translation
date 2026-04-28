from dataclasses import dataclass
from typing import Literal

from anthropic import Anthropic
from openai import OpenAI

TranslationProvider = Literal["chatgpt", "claude", "internal_llm"]

SYSTEM_PROMPT = (
    "You are a professional academic translator specializing in technical and scientific papers. "
    "Translate the given text into Korean. "
    "Rules:\n"
    "- Preserve all technical terms, proper nouns, and abbreviations in their original form (or add Korean in parentheses).\n"
    "- Maintain the original tone, structure, and paragraph breaks.\n"
    "- Do NOT add explanations, summaries, or commentary.\n"
    "- Output ONLY the translated Korean text."
)


@dataclass
class TranslationConfig:
    provider: TranslationProvider
    api_key: str
    model: str
    base_url: str | None = None


def translate_text(config: TranslationConfig, text: str) -> str:
    if not text.strip():
        return text
    if config.provider == "claude":
        return _translate_claude(config, text)
    return _translate_openai_compatible(config, text)


def _translate_openai_compatible(config: TranslationConfig, text: str) -> str:
    kwargs: dict = {"api_key": config.api_key}
    if config.provider == "internal_llm":
        if not config.base_url:
            raise ValueError("내부 LLM은 base_url이 필요합니다.")
        kwargs["base_url"] = config.base_url.rstrip("/")
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _translate_claude(config: TranslationConfig, text: str) -> str:
    client = Anthropic(api_key=config.api_key)
    message = client.messages.create(
        model=config.model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
        temperature=0.2,
    )
    block = message.content[0]
    if block.type != "text":
        return ""
    return block.text.strip()
