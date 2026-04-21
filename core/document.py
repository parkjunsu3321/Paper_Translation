import base64
from io import BytesIO
from typing import Callable

from docling_core.types.doc import DocItemLabel, TableItem, PictureItem
from openai import OpenAI

from .styles import CSS
from .translator import translate_text


HEADING_TAGS = {
    DocItemLabel.TITLE: "h1",
    DocItemLabel.SECTION_HEADER: "h2",
}

SKIP_LABELS = {DocItemLabel.TABLE, DocItemLabel.PICTURE}


def picture_to_base64(item: PictureItem, doc) -> str | None:
    try:
        image = item.get_image(doc)
        if image is None:
            return None
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def table_to_html(item: TableItem) -> str:
    try:
        df = item.export_to_dataframe()
        return df.to_html(index=False, border=0, classes="docling-table")
    except Exception:
        try:
            return item.export_to_html()
        except Exception:
            return "<p>[표]</p>"


def build_html(
    doc,
    client: OpenAI,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> str:
    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        f"<style>{CSS}</style>",
        "</head><body>",
    ]

    items = list(doc.iterate_items())
    text_items = [
        (item, lv) for item, lv in items
        if hasattr(item, "text") and item.text.strip()
        and getattr(item, "label", None) not in SKIP_LABELS
    ]
    total = len(text_items)
    translated = 0

    for item, _level in items:
        label = getattr(item, "label", None)

        if isinstance(item, TableItem):
            parts.append(table_to_html(item))

        elif isinstance(item, PictureItem):
            b64 = picture_to_base64(item, doc)
            if b64:
                parts.append(f'<img src="data:image/png;base64,{b64}" alt="figure" />')
            else:
                parts.append("<p>[그림]</p>")

        elif hasattr(item, "text") and item.text.strip():
            tag = HEADING_TAGS.get(label, "p")
            translated += 1
            if on_progress:
                on_progress(translated, total, item.text[:50].replace("\n", " "))

            korean = translate_text(client, item.text)
            korean = korean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f"<{tag}>{korean}</{tag}>")

    parts.append("</body></html>")
    return "\n".join(parts)
