import streamlit as st
import pandas as pd
from pathlib import Path
import time
import json

# Import the refactored functions from the modules
from modules import scheme_processing, ocr_processing, scoring

# --- 1. SETUP: DIRECTORIES AND PATHS ---
APP_DIR = Path.cwd()
UPLOADS_DIR = APP_DIR / "data" / "uploads"
PROCESSED_DIR = APP_DIR / "data" / "processed"
SCHEME_ARTIFACTS_DIR = PROCESSED_DIR / "scheme_artifacts"
STUDENT_ARTIFACTS_DIR = PROCESSED_DIR / "student_artifacts"
CROPPED_IMAGES_DIR = STUDENT_ARTIFACTS_DIR / "cropped_images"
RESULTS_DIR = PROCESSED_DIR / "results"

# --- 2. STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="Automated Answer Grader", page_icon="🤖", layout="wide")
st.title("🤖 Automated Answer Grader")
st.markdown("An intelligent system to parse, evaluate, and score student answers using AI.")

# --- Helper function for styling the results dataframe ---
def style_confidence(val):
    try:
        v = float(val)
        if v >= 0.85: color = 'rgba(42, 187, 155, 0.3)'  # Green
        elif v >= 0.6: color = 'rgba(255, 193, 7, 0.3)'   # Yellow
        else: color = 'rgba(239, 83, 80, 0.3)'            # Red
        return f'background-color: {color}'
    except (ValueError, TypeError):
        return ''

# --- 3. SIDEBAR FOR FILE UPLOADS AND CONTROLS ---
with st.sidebar:
    st.header("Setup")
    
    st.write("Initializing data directories...")
    try:
        for path in [UPLOADS_DIR, SCHEME_ARTIFACTS_DIR, STUDENT_ARTIFACTS_DIR, CROPPED_IMAGES_DIR, RESULTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        st.write("✅ Directories are ready.")
    except Exception as e:
        st.error(f"Failed to create directories: {e}")
    
    st.markdown("---")
    st.markdown("Upload files, or place them in the `data/uploads` folder and refresh.")
    
    # Check for existing files
    existing_scheme_pdf = next(UPLOADS_DIR.glob('*scheme*.pdf'), None)
    existing_student_pdf = next(UPLOADS_DIR.glob('*answers*.pdf'), None)
    existing_coords_json = next(UPLOADS_DIR.glob('*coords*.json'), None)

    if existing_scheme_pdf and existing_student_pdf and existing_coords_json:
        st.success("Found existing files! Ready to grade.")

    scheme_pdf_file = st.file_uploader("1. Upload Marking Scheme PDF", type="pdf")
    student_answers_pdf_file = st.file_uploader("2. Upload Student Answers PDF", type="pdf")
    coords_json_file = st.file_uploader("3. Upload Coordinates JSON", type="json")
    st.markdown("---")
    
    # Enable button if new files are uploaded OR if existing files are found
    can_start = (scheme_pdf_file and student_answers_pdf_file and coords_json_file) or \
                (existing_scheme_pdf and existing_student_pdf and existing_coords_json)

    start_button = st.button("Start Grading", type="primary", disabled=not can_start, use_container_width=True)
    if not can_start:
        st.warning("Please upload all required files to enable grading.")
    
    st.markdown("---")
    st.header("Advanced")
    if st.button("Clear Cache and Reset", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! The app is reset.")

# --- 4. MAIN WORKFLOW ORCHESTRATION ---
if start_button:
    main_status = st.status("Running the grading pipeline...", expanded=True)
    try:
        # --- A. Determine file paths (use uploaded files first, then existing) ---
        main_status.write("Determining file paths...")
        
        if scheme_pdf_file:
            scheme_pdf_path = UPLOADS_DIR / scheme_pdf_file.name
            with open(scheme_pdf_path, "wb") as f: f.write(scheme_pdf_file.getbuffer())
        else:
            scheme_pdf_path = existing_scheme_pdf

        if student_answers_pdf_file:
            student_pdf_path = UPLOADS_DIR / student_answers_pdf_file.name
            with open(student_pdf_path, "wb") as f: f.write(student_answers_pdf_file.getbuffer())
        else:
            student_pdf_path = existing_student_pdf

        if coords_json_file:
            coords_json_path = UPLOADS_DIR / coords_json_file.name
            with open(coords_json_path, "wb") as f: f.write(coords_json_file.getbuffer())
        else:
            coords_json_path = existing_coords_json
        
        st.write(f"Using Marking Scheme: `{scheme_pdf_path.name}`")
        st.write(f"Using Student Answers: `{student_pdf_path.name}`")
        st.write(f"Using Coordinates: `{coords_json_path.name}`")
        time.sleep(1)

        # --- B. Execute the Grading Pipeline ---
        main_status.write("Step 1/3: Processing the marking scheme...")
        raw_scheme_text = scheme_processing.process_scheme(scheme_pdf_path, SCHEME_ARTIFACTS_DIR)
        st.write("✅ Marking scheme parsed and vector index built.")
        
        with st.expander("Sanity Check: Review Parsed Marking Scheme"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Parsed Scheme Data")
                meta_path = SCHEME_ARTIFACTS_DIR / "scheme_meta.jsonl"
                if meta_path.exists():
                    points = [json.loads(line) for line in open(meta_path, 'r', encoding='utf8')]
                    scheme_df = pd.DataFrame(points)
                    st.dataframe(scheme_df, use_container_width=True, height=300)
            with col2:
                st.subheader("Raw Extracted Text from PDF")
                st.text_area("Raw Text", raw_scheme_text, height=300)
        
        time.sleep(1)

        main_status.write("Step 2/3: Processing student answers (Cropping & OCR)...")
        student_csv_path = STUDENT_ARTIFACTS_DIR / "students_ocr.csv"
        ocr_processing.process_student_pdf(student_pdf_path, coords_json_path, STUDENT_ARTIFACTS_DIR, student_csv_path)
        st.write("✅ Student answers PDF cropped and text extracted via OCR.")

        with st.expander("Sanity Check: Review Extracted Student Answers"):
            if student_csv_path.exists():
                student_df = pd.read_csv(student_csv_path)
                st.dataframe(student_df, use_container_width=True)
        
        time.sleep(1)

        main_status.write(f"Step 3/3: Scoring answers with AI...")
        model_info_placeholder = st.empty()
        final_scores_path = RESULTS_DIR / "final_scores.csv"
        scoring.score_answers(student_csv_path, SCHEME_ARTIFACTS_DIR, final_scores_path, model_info_placeholder)
        st.write("✅ LLM scoring complete.")
        
        main_status.update(label="Grading Pipeline Complete!", state="complete", expanded=False)
        
        # --- C. Display the Results ---
        st.header("📊 Grading Results")
        if final_scores_path.exists():
            results_df = pd.read_csv(final_scores_path)
            st.dataframe(results_df.style.applymap(style_confidence, subset=['confidence_score']), use_container_width=True)
            with open(final_scores_path, "rb") as file:
                st.download_button("Download Scores as CSV", file, "final_scores.csv", "text/csv", use_container_width=True)
        else:
            st.warning("Could not find the final scores file.")

    except Exception as e:
        main_status.update(label="An error occurred!", state="error")
        st.error(f"An error occurred during the pipeline: {e}")
        st.exception(e)
else:
    st.info("Please upload your files, or place them in the `data/uploads` folder and refresh the page.")
