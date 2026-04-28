import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.pipeline import run_translation
from core.translator import TranslationConfig, TranslationProvider

load_dotenv()


def _default_model(provider: TranslationProvider) -> str:
    env_map = {
        "chatgpt": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        "claude": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        "internal_llm": os.environ.get("INTERNAL_LLM_MODEL", "llama3.1"),
    }
    return env_map[provider]


def _make_progress_callback(progress_bar, status_text):
    def on_progress(done: int, total: int, preview: str):
        status_text.text(f"번역 중... [{done}/{total}]  {preview}...")
        progress_bar.progress(done / total)
    return on_progress


st.set_page_config(page_title="논문 번역기", page_icon="📄", layout="centered")
st.title("📄 논문 번역기")
st.caption("PDF를 한국어로 번역합니다. 이미지·표는 원본 그대로 유지됩니다.")

provider_labels: dict[str, TranslationProvider] = {
    "ChatGPT (OpenAI)": "chatgpt",
    "Claude (Anthropic)": "claude",
    "내부 LLM (OpenAI 호환 API)": "internal_llm",
}

default_openai = os.environ.get("OPENAI_API_KEY", "")
default_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
default_internal_key = os.environ.get("INTERNAL_LLM_API_KEY", "")
default_internal_base = os.environ.get("INTERNAL_LLM_BASE_URL", "http://localhost:11434/v1")

with st.sidebar:
    st.header("설정")
    provider_choice = st.selectbox(
        "번역 엔진",
        list(provider_labels.keys()),
        help="OpenAI, Anthropic, 또는 사내/로컬 등 OpenAI Chat Completions 호환 엔드포인트를 선택합니다.",
    )
    provider = provider_labels[provider_choice]

    model = st.text_input(
        "모델 ID",
        value=_default_model(provider),
        key=f"model_id_{provider}",
        help="예: gpt-4o, claude-3-5-sonnet-20241022, Ollama의 모델명 등",
    )

    api_key = ""
    base_url: str | None = None

    if provider == "chatgpt":
        api_key = st.text_input(
            "OpenAI API 키",
            value=default_openai,
            type="password",
            placeholder="sk-...",
            help=".env의 OPENAI_API_KEY가 있으면 자동으로 채워집니다.",
        )
    elif provider == "claude":
        api_key = st.text_input(
            "Anthropic API 키",
            value=default_anthropic,
            type="password",
            placeholder="sk-ant-...",
            help=".env의 ANTHROPIC_API_KEY가 있으면 자동으로 채워집니다.",
        )
    else:
        base_url = st.text_input(
            "Base URL",
            value=default_internal_base,
            placeholder="https://api.example.com/v1",
            help="OpenAI SDK 호환 베이스 URL (끝에 /v1 포함). Ollama 예: http://localhost:11434/v1",
        )
        api_key = st.text_input(
            "API 키 (선택)",
            value=default_internal_key,
            type="password",
            help="로컬 Ollama 등은 임의 문자열을 넣어도 됩니다. 비우면 ollama로 전송합니다.",
        )

input_mode = st.radio("입력 방식", ["파일 업로드", "URL 입력"], horizontal=True)

source = None
uploaded_tmp = None

if input_mode == "파일 업로드":
    uploaded = st.file_uploader("PDF 파일 선택", type=["pdf"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded.read())
        tmp.flush()
        uploaded_tmp = tmp.name
        source = uploaded_tmp
        st.success(f"업로드됨: {uploaded.name}")
else:
    url = st.text_input("PDF URL 입력", placeholder="https://arxiv.org/pdf/...")
    if url.strip():
        source = url.strip()

internal_ready = provider != "internal_llm" or bool((base_url or "").strip())

if provider == "internal_llm":
    key_ready = internal_ready
else:
    key_ready = bool(api_key.strip())

disabled = not source or not key_ready or not model.strip()

if st.button("번역 시작", type="primary", disabled=disabled):
    progress_bar = st.progress(0)
    status_text = st.empty()

    internal_key = (api_key or "").strip() or "ollama"
    internal_base = ((base_url or "").strip() or None) if provider == "internal_llm" else None
    config = TranslationConfig(
        provider=provider,
        api_key=api_key.strip() if provider != "internal_llm" else internal_key,
        model=model.strip(),
        base_url=internal_base,
    )

    try:
        pdf_bytes, html = run_translation(
            source,
            config,
            on_status=status_text.text,
            on_progress=_make_progress_callback(progress_bar, status_text),
        )

        progress_bar.progress(1.0)
        status_text.text("✅ 번역 완료!")
        st.success("번역이 완료되었습니다.")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 번역 PDF 다운로드",
                data=pdf_bytes,
                file_name="translated_output.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📥 HTML 다운로드",
                data=html.encode("utf-8"),
                file_name="translated_output.html",
                mime="text/html",
                use_container_width=True,
            )

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"오류 발생: {e}")

    finally:
        if uploaded_tmp and os.path.exists(uploaded_tmp):
            os.unlink(uploaded_tmp)
