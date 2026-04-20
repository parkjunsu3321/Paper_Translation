import torch
import os
import fitz
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from tqdm import tqdm
from config import Config

config = Config()


def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


class TableExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained(config.TABLE_DET_MODEL)
        self.model = TableTransformerForObjectDetection.from_pretrained(
            config.TABLE_DET_MODEL
        ).to(self.device)

    def extract_tables(self, pdf_path, output_folder="extracted_tables", padding=20):
        os.makedirs(output_folder, exist_ok=True)

        images = pdf_to_images(pdf_path, dpi=300)

        for page_idx, image in enumerate(tqdm(images, desc="Extracting Tables")):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)

            results = self.processor.post_process_object_detection(
                outputs, threshold=0.98, target_sizes=target_sizes
            )[0]

            for table_idx, box in enumerate(results["boxes"]):
                x_min, y_min, x_max, y_max = box.int().tolist()

                # 패딩 적용 (이미지 경계 초과 방지)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.width, x_max + padding)
                y_max = min(image.height, y_max + padding)

                table_crop = image.crop((x_min, y_min, x_max, y_max))

                filename = f"page_{page_idx + 1}_table_{table_idx + 1}.png"
                table_crop.save(os.path.join(output_folder, filename))