from typing import Callable

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from openai import OpenAI
from weasyprint import HTML as WeasyHTML

from .document import build_html


def run_translation(
    source: str,
    api_key: str,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[bytes, str]:
    client = OpenAI(api_key=api_key)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if on_status:
        on_status("문서 파싱 중...")
    result = converter.convert(source)
    doc = result.document

    if on_status:
        on_status("텍스트 번역 중 (이미지·표는 유지)...")
    html = build_html(doc, client, on_progress=on_progress)

    if on_status:
        on_status("PDF 생성 중...")
    pdf_bytes = WeasyHTML(string=html, base_url=".").write_pdf()

    return pdf_bytes, html
