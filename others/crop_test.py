import json
import cv2
import numpy as np
import fitz  # PyMuPDF
import os

PDF_PATH = "student_sample.pdf"
COORDS_PATH = "box_coords.json"
OUTPUT_DIR = "cropped_test"
DPI = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load coords
with open(COORDS_PATH, "r") as f:
    coords = json.load(f)

# Open PDF with fitz
doc = fitz.open(PDF_PATH)

# Render first page as image
page = doc[0]  # first page
pix = page.get_pixmap(dpi=DPI)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# Crop each region
for q_num, regions in coords.items():
    for region_type, (pt1, pt2) in regions.items():
        x1, y1 = pt1
        x2, y2 = pt2
        cropped = img[y1:y2, x1:x2]

        out_path = os.path.join(OUTPUT_DIR, f"Q{q_num}_{region_type}.png")
        cv2.imwrite(out_path, cropped)
        print(f"💾 Saved {out_path}")

print("✅ Cropping done using PyMuPDF.")
