import os
import fitz
from tqdm import tqdm


class ImageExtractor:
    def extract_images(self, pdf_path, output_folder="extracted_images"):
        os.makedirs(output_folder, exist_ok=True)
        doc = fitz.open(pdf_path)

        for page_idx in tqdm(range(len(doc)), desc="Extracting Images"):
            page = doc[page_idx]
            image_list = page.get_images(full=True)

            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                img_filename = f"page_{page_idx + 1}_img_{img_idx + 1}.{image_ext}"
                with open(os.path.join(output_folder, img_filename), "wb") as f:
                    f.write(image_bytes)
