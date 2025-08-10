import json
import re
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import streamlit as st

# --- Part 1: Functions from parse_scheme.py ---

def _extract_text_from_pdf(pdf_path):
    """Extracts all text from a given PDF file."""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            page_text = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text.append(page_text)
    return "\n\n".join(text)

def _split_questions(full_text):
    """Splits the full text into blocks based on question numbers."""
    parts = re.split(r'(?m)^\s*(\d{1,3})\b', full_text)
    items = []
    for i in range(1, len(parts), 2):
        qid = parts[i].strip()
        body = parts[i+1].strip()
        items.append((qid, body))
    return items

def _split_points(qbody):
    """Splits a question body into individual marking points."""
    bullets = re.split(r'\(\s*[ivxlcdm]+\s*\)', qbody, flags=re.I)
    pts = [b.strip() for b in bullets if b.strip()]
    return pts

def _detect_max_marks(qbody):
    """Detects mark allocation patterns like '3X1=3'."""
    m = re.search(r'(\d+)\s*[xX]\s*(\d+)\s*=\s*(\d+)', qbody)
    if m:
        total, parts, per = int(m.group(3)), int(m.group(1)), int(m.group(2))
        return total, parts, per
    return None

# --- Part 2: Functions from build_index.py ---

def _load_points(path):
    """Loads points from a JSONL file."""
    pts = []
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            pts.append(json.loads(line))
    return pts

@st.cache_resource
def _get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Loads and caches the sentence transformer model."""
    return SentenceTransformer(model_name)

# --- Main Orchestration Function ---

def process_scheme(pdf_path, output_dir: Path):
    """
    Main function to process the marking scheme PDF.
    It parses the PDF, creates embeddings, and saves the FAISS index and metadata.
    """
    st.write("-> Extracting text from marking scheme PDF...")
    full_text = _extract_text_from_pdf(pdf_path)
    
    st.write("-> Splitting text into questions and points...")
    questions = _split_questions(full_text)
    
    scheme_points_path = output_dir / "scheme_points.jsonl"
    with scheme_points_path.open("w", encoding="utf8") as fh:
        for qid, body in tqdm(questions, desc="Parsing Questions"):
            pts = _split_points(body)
            mark_info = _detect_max_marks(body)
            per_point = 1
            if mark_info:
                _, _, per = mark_info
                per_point = per if per > 0 else 1
            
            for idx, pt_text in enumerate(pts, start=1):
                obj = {
                    "question_id": str(qid),
                    "point_index": idx,
                    "text": re.sub(r'\s+', ' ', pt_text).strip(),
                    "marks": per_point
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    st.write(f"-> Saved {len(questions)} questions to scheme_points.jsonl.")

    # Now, build the index from the generated file
    st.write("-> Building vector index from scheme points...")
    points = _load_points(scheme_points_path)
    
    model = _get_embedding_model()
    
    texts = [p["text"] for p in points]
    
    st.write("-> Encoding texts... (This might take a moment on first run)")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))
    
    # Define output paths
    index_path = output_dir / "scheme.index"
    emb_path = output_dir / "scheme_embeddings.npy"
    meta_path = output_dir / "scheme_meta.jsonl"
    
    faiss.write_index(index, str(index_path))
    np.save(emb_path, embeddings.astype("float32"))
    
    with open(meta_path, 'w', encoding='utf8') as fh:
        for i, meta in enumerate(points):
            meta_rec = {
                "fid": i,
                "question_id": meta["question_id"],
                "point_index": meta["point_index"],
                "text": meta["text"],
                "marks": meta.get("marks", 1)
            }
            fh.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")
            
    st.write("-> FAISS index, embeddings, and metadata saved successfully.")
