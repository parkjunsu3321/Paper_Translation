"""
PDF → Markdown 변환기

- 표 탐지: microsoft/table-transformer-detection (DetrImageProcessor)
- 표 구조: microsoft/table-transformer-structure-recognition → GFM 테이블
- 이미지:  fitz get_images/extract_image → PNG 저장 → ![...](images/...)
- 텍스트:  pdfplumber within_bbox로 표·이미지 제외 영역 추출
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import fitz
import pdfplumber
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1  (PDF points)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent
PDF_INPUT  = _BASE / "paper.pdf"
OUTPUT_MD  = _BASE / "parsed.md"
IMAGE_DIR  = _BASE / "images"

TABLE_DET_MODEL = "microsoft/table-transformer-detection"
TABLE_STR_MODEL = "microsoft/table-transformer-structure-recognition"

RENDER_DPI          = 300
TABLE_DET_THRESHOLD = 0.8
TABLE_STR_THRESHOLD = 0.5
TABLE_PADDING_PT    = 5.0   # 탐지된 표 bbox에 추가할 여백 (PDF pt)
MIN_IMAGE_PT        = 40.0  # 이 크기 미만 이미지는 장식으로 무시
STRIP_REFERENCES    = True


# ---------------------------------------------------------------------------
# 모델 로딩 (공유 캐시)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_model(model_id: str, device_str: str):
    device = torch.device(device_str)
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    model.to(device).eval()
    return model, processor, device


# ---------------------------------------------------------------------------
# 표 탐지
# ---------------------------------------------------------------------------

class TableDetector:
    def __init__(self, model_id: str = TABLE_DET_MODEL, device: str | None = None):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.device = _load_model(model_id, dev)

    def detect(
        self,
        image: Image.Image,
        page_w_pt: float,
        page_h_pt: float,
        *,
        threshold: float = TABLE_DET_THRESHOLD,
        padding_pt: float = TABLE_PADDING_PT,
    ) -> List[BBox]:
        """PIL 이미지에서 표를 탐지해 PDF 포인트 좌표 bbox 리스트로 반환."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        h, w = image.height, image.width
        target_sizes = torch.tensor([[h, w]], device=self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        sx, sy = page_w_pt / w, page_h_pt / h
        boxes: List[BBox] = []
        for label, box in zip(results["labels"], results["boxes"]):
            if "table" not in self.model.config.id2label.get(int(label), "").lower():
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            boxes.append((
                max(0.0,       xmin * sx - padding_pt),
                max(0.0,       ymin * sy - padding_pt),
                min(page_w_pt, xmax * sx + padding_pt),
                min(page_h_pt, ymax * sy + padding_pt),
            ))
        return boxes


# ---------------------------------------------------------------------------
# 표 구조 인식 → 셀 텍스트
# ---------------------------------------------------------------------------

class TableStructureRecognizer:
    def __init__(self, model_id: str = TABLE_STR_MODEL, device: str | None = None):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor, self.device = _load_model(model_id, dev)

    def to_rows(
        self,
        table_pil: Image.Image,
        table_pdf_bbox: BBox,
        page: pdfplumber.page.Page,
        *,
        threshold: float = TABLE_STR_THRESHOLD,
    ) -> List[List[str]]:
        """표 이미지 → 행×열 셀 텍스트 2D 리스트."""
        if table_pil.width < 4 or table_pil.height < 4:
            return []

        inputs = self.processor(images=table_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        h, w = table_pil.height, table_pil.width
        target_sizes = torch.tensor([[h, w]], device=self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        row_boxes, col_boxes = [], []
        for label, box in zip(results["labels"], results["boxes"]):
            name = self.model.config.id2label.get(int(label), "").lower()
            b = box.tolist()
            if "row" in name:
                row_boxes.append(b)
            elif "column" in name:
                col_boxes.append(b)

        if not row_boxes or not col_boxes:
            return []

        row_boxes.sort(key=lambda b: b[1])
        col_boxes.sort(key=lambda b: b[0])

        tx0, ty0, tx1, ty1 = table_pdf_bbox
        sx = max(tx1 - tx0, 1.0) / max(w, 1)
        sy = max(ty1 - ty0, 1.0) / max(h, 1)

        def to_pdf(xmin, ymin, xmax, ymax) -> BBox:
            return (tx0 + xmin * sx, ty0 + ymin * sy, tx0 + xmax * sx, ty0 + ymax * sy)

        table_data: List[List[str]] = []
        for row in row_boxes:
            row_data: List[str] = []
            for col in col_boxes:
                cx0 = max(row[0], col[0])
                cy0 = max(row[1], col[1])
                cx1 = min(row[2], col[2])
                cy1 = min(row[3], col[3])
                if cx0 >= cx1 or cy0 >= cy1:
                    row_data.append("")
                    continue
                try:
                    cell_text = (
                        page.within_bbox(to_pdf(cx0, cy0, cx1, cy1))
                        .extract_text(x_tolerance=2, y_tolerance=2) or ""
                    ).strip()
                except Exception:
                    cell_text = ""
                row_data.append(re.sub(r"\s+", " ", cell_text))
            if any(row_data):
                table_data.append(row_data)

        return table_data


# ---------------------------------------------------------------------------
# 이미지 추출 (fitz xref 기반, parser/img_extraction.py 방식)
# ---------------------------------------------------------------------------

class ImageExtractor:
    def extract_page(
        self,
        fitz_doc: fitz.Document,
        page_idx: int,
        output_dir: Path,
        *,
        min_pt: float = MIN_IMAGE_PT,
    ) -> List[tuple[Path, BBox]]:
        """페이지 임베디드 이미지를 저장하고 (파일경로, pdf_bbox) 리스트 반환."""
        page = fitz_doc[page_idx]
        results = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                rect = page.get_image_bbox(img)
            except Exception:
                continue
            if (rect.x1 - rect.x0) < min_pt or (rect.y1 - rect.y0) < min_pt:
                continue
            base_img = fitz_doc.extract_image(xref)
            ext = base_img.get("ext", "png")
            fname = f"p{page_idx + 1}_img{img_idx + 1}.{ext}"
            out_path = output_dir / fname
            out_path.write_bytes(base_img["image"])
            results.append((out_path, (rect.x0, rect.y0, rect.x1, rect.y1)))
        return results


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _page_to_pil(fitz_page: fitz.Page, dpi: int = RENDER_DPI) -> Image.Image:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _crop_pil(pil: Image.Image, bbox_pdf: BBox, page_w_pt: float, page_h_pt: float) -> Image.Image:
    sx, sy = pil.width / page_w_pt, pil.height / page_h_pt
    x0, y0, x1, y1 = bbox_pdf
    return pil.crop((
        max(0, int(x0 * sx)), max(0, int(y0 * sy)),
        min(pil.width, int(x1 * sx)), min(pil.height, int(y1 * sy)),
    ))


def _table_to_markdown(rows: List[List]) -> str:
    if not rows:
        return ""
    cleaned = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in row] for row in rows]
    n_cols = max((len(r) for r in cleaned), default=0)
    if not n_cols:
        return ""
    padded = [r + [""] * (n_cols - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(padded[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def _band_text(page: pdfplumber.page.Page, y0: float, y1: float) -> str:
    if y1 <= y0 + 1:
        return ""
    try:
        return (
            page.within_bbox((0, y0, float(page.width), y1))
            .extract_text(x_tolerance=2, y_tolerance=2) or ""
        ).strip()
    except Exception:
        return ""


def _plumber_fallback(page: pdfplumber.page.Page, bbox: BBox) -> List[List[str]]:
    """pdfplumber로 표 셀 추출 (structure 인식 실패 시 폴백)."""
    x0, y0, x1, y1 = bbox
    matched = next(
        (t for t in (page.find_tables() or [])
         if not (t.bbox[2] < x0 or x1 < t.bbox[0] or t.bbox[3] < y0 or y1 < t.bbox[1])),
        None,
    )
    if matched:
        rows = matched.extract()
        if rows:
            return rows
    for strategy in ("lines_strict", "text"):
        try:
            rows = page.within_bbox(bbox).extract_table(
                table_settings={
                    "vertical_strategy": strategy,
                    "horizontal_strategy": strategy,
                    "snap_tolerance": 3,
                }
            ) or []
            if rows:
                return rows
        except Exception:
            pass
    return []


def _strip_references(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*$", line.strip()):
            return "\n".join(lines[:i]).rstrip()
    return text


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str | Path,
    *,
    image_dir: Path | None = None,
    device: str | None = None,
    render_dpi: int = RENDER_DPI,
    table_det_threshold: float = TABLE_DET_THRESHOLD,
    table_str_threshold: float = TABLE_STR_THRESHOLD,
    strip_references: bool = STRIP_REFERENCES,
) -> str:
    """PDF 전체를 Markdown 문자열로 반환한다."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    _image_dir = image_dir or IMAGE_DIR
    _image_dir.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector   = TableDetector(device=dev)
    recognizer = TableStructureRecognizer(device=dev)
    extractor  = ImageExtractor()

    fitz_doc = fitz.open(str(path))
    page_mds: List[str] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="페이지 처리")):
            pw, ph = float(page.width), float(page.height)
            pil = _page_to_pil(fitz_doc[i], dpi=render_dpi)

            table_bboxes = detector.detect(pil, pw, ph, threshold=table_det_threshold)
            print(f"  p{i+1}: 표 탐지 {len(table_bboxes)}개")
            img_items    = extractor.extract_page(fitz_doc, i, _image_dir)

            # elements: (y_top, y_bottom, kind, data)
            elements = []

            for t_idx, bbox in enumerate(table_bboxes):
                table_pil = _crop_pil(pil, bbox, pw, ph)

                # 표 이미지 저장
                tbl_fname = f"p{i + 1}_table{t_idx + 1}.png"
                tbl_img_path = _image_dir / tbl_fname
                table_pil.save(tbl_img_path, "PNG")
                tbl_img_rel = str(tbl_img_path.relative_to(_BASE)).replace("\\", "/")

                rows = recognizer.to_rows(table_pil, bbox, page, threshold=table_str_threshold)
                src = "structure"
                if not rows:
                    rows = _plumber_fallback(page, bbox)
                    src = "plumber"
                md_tbl = _table_to_markdown(rows)
                if md_tbl:
                    print(f"    표{t_idx+1}: {len(rows)}행 ({src})")
                    # 마크다운 테이블 + 표 이미지 참조 함께 저장
                    combined = f"![표 {t_idx+1} (페이지 {i+1})]({tbl_img_rel})\n\n{md_tbl}"
                    elements.append((bbox[1], bbox[3], "table", combined))
                else:
                    print(f"    표{t_idx+1}: 셀 추출 실패 → 이미지만 저장")
                    elements.append((bbox[1], bbox[3], "table", f"![표 {t_idx+1} (페이지 {i+1})]({tbl_img_rel})"))

            for out_path, img_bbox in img_items:
                rel = str(out_path.relative_to(_BASE)).replace("\\", "/")
                elements.append((img_bbox[1], img_bbox[3], "image", rel))

            elements.sort(key=lambda e: e[0])

            parts: List[str] = []
            prev_y = 0.0
            img_counter = 0

            for y_top, y_bottom, kind, data in elements:
                txt = _band_text(page, prev_y, y_top)
                if txt:
                    parts.append(txt)
                if kind == "table":
                    parts.append(data)
                else:
                    img_counter += 1
                    parts.append(f"![그림 {img_counter} (페이지 {i + 1})]({data})")
                prev_y = max(prev_y, y_bottom)

            tail = _band_text(page, prev_y, ph)
            if tail:
                parts.append(tail)

            if not elements:
                full = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
                if full:
                    page_mds.append(full)
            elif parts:
                page_mds.append("\n\n".join(p for p in parts if p))

    fitz_doc.close()

    text = "\n\n---\n\n".join(page_mds)
    if strip_references:
        text = _strip_references(text)
    return text


# ---------------------------------------------------------------------------
# 엔트리포인트
# ---------------------------------------------------------------------------

def main() -> None:
    if not PDF_INPUT.is_file():
        raise FileNotFoundError(f"PDF 없음: {PDF_INPUT}")

    md = parse_pdf(PDF_INPUT, image_dir=IMAGE_DIR)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"저장: {OUTPUT_MD} ({len(md):,} chars)")
    img_count = len(list(IMAGE_DIR.glob("*"))) if IMAGE_DIR.exists() else 0
    print(f"이미지: {IMAGE_DIR} ({img_count}개)")


if __name__ == "__main__":
    main()
