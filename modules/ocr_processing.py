import json
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import re
import csv
from pathlib import Path
import easyocr
import streamlit as st

# --- Part 1: Functions from save_cropped_images.py ---

def _crop_and_save_images(pdf_path: Path, coords_path: Path, output_dir: Path):
    """
    Crops a PDF based on JSON coordinates and saves the images.
    """
    # Ensure the output directory for cropped images exists
    cropped_images_dir = output_dir / "cropped_images"
    cropped_images_dir.mkdir(exist_ok=True)
    
    with open(coords_path, "r") as f:
        coords = json.load(f)

    doc = fitz.open(pdf_path)
    
    # Process only the first page as per the original script's logic
    if doc.page_count > 0:
        page_num = 0
        page = doc[page_num]
        
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3: # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for q_num, regions in coords.items():
            for region_type, (pt1, pt2) in regions.items():
                x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
                y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
                
                cropped_img = img[y_min:y_max, x_min:x_max]
                
                crop_filename = f"Pair{q_num}_{region_type}_page{page_num}.png"
                crop_path = cropped_images_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_img)
    
    doc.close()
    return cropped_images_dir

# --- Part 2: Functions from printed_ocr.py ---

@st.cache_resource
def _get_ocr_reader():
    """Loads and caches the EasyOCR reader."""
    return easyocr.Reader(['en'], gpu=False)

def _run_ocr_on_images(images_dir: Path, output_csv_path: Path):
    """
    Runs OCR on the cropped images and saves the results to a CSV file.
    """
    reader = _get_ocr_reader()
    
    # Find all cropped image pairs
    image_files = os.listdir(images_dir)
    question_ids = sorted(list(set([re.search(r'Pair(\d+)', f).group(1) for f in image_files if re.search(r'Pair(\d+)', f)])))

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['student_id', 'question_id', 'answer_text'])
        
        student_id = 'S_01' # Default student ID for this run

        for q_id in question_ids:
            num_file = images_dir / f"Pair{q_id}_number_page0.png"
            ans_file = images_dir / f"Pair{q_id}_answer_page0.png"
            
            if num_file.exists() and ans_file.exists():
                try:
                    # OCR for question number (though we already have it)
                    # This part could be simplified if q_id is trusted
                    number_results = reader.readtext(str(num_file))
                    full_number_text = ' '.join([res[1] for res in number_results])
                    match = re.search(r'\d+', full_number_text)
                    question_id_ocr = match.group(0) if match else q_id

                    # OCR for answer text
                    answer_results = reader.readtext(str(ans_file), paragraph=True)
                    answer_text = ' '.join([res[1] for res in answer_results])

                    csv_writer.writerow([student_id, question_id_ocr, answer_text])
                except Exception as e:
                    st.warning(f"Could not process OCR for Question ID {q_id}: {e}")


# --- Main Orchestration Function ---

def process_student_pdf(pdf_path: Path, coords_path: Path, student_artifacts_dir: Path, output_csv_path: Path):
    """
    Main function to process the student answer PDF.
    It crops the PDF into images and then runs OCR on them.
    """
    st.write("-> Cropping student answer sheet...")
    cropped_images_dir = _crop_and_save_images(pdf_path, coords_path, student_artifacts_dir)
    st.write(f"-> Saved cropped images to `{cropped_images_dir}`")
    
    st.write("-> Running OCR on cropped images...")
    _run_ocr_on_images(cropped_images_dir, output_csv_path)
    st.write(f"-> Saved OCR results to `{output_csv_path}`")

