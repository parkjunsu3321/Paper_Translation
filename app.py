import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.pipeline import run_translation

load_dotenv()


def _make_progress_callback(progress_bar, status_text):
    def on_progress(done: int, total: int, preview: str):
        status_text.text(f"번역 중... [{done}/{total}]  {preview}...")
        progress_bar.progress(done / total)
    return on_progress


st.set_page_config(page_title="논문 번역기", page_icon="📄", layout="centered")
st.title("📄 논문 번역기")
st.caption("PDF를 한국어로 번역합니다. 이미지·표는 원본 그대로 유지됩니다.")

default_api_key = os.environ.get("OPENAI_API_KEY", "")
with st.sidebar:
    st.header("설정")
    api_key = st.text_input(
        "OpenAI API 키",
        value=default_api_key,
        type="password",
        placeholder="sk-...",
        help=".env 파일에 OPENAI_API_KEY가 있으면 자동으로 채워집니다.",
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

if st.button("번역 시작", type="primary", disabled=not source or not api_key):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        pdf_bytes, html = run_translation(
            source,
            api_key,
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
