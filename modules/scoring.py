import json
import pandas as pd
import subprocess
from pathlib import Path
import streamlit as st
import re

# --- Helper Functions ---

def _load_meta(meta_path):
    """Loads metadata from a JSONL file."""
    meta = []
    with open(meta_path, 'r', encoding='utf8') as fh:
        for line in fh:
            meta.append(json.loads(line))
    return meta

def _ollama_chat_with_fallback(model_list, model_info_placeholder, prompt):
    """
    Tries to run Ollama with a list of models, falling back if a memory error occurs.
    """
    for model_name in model_list:
        try:
            model_info_placeholder.info(f"Attempting to use model: `{model_name}`...")
            cmd = ["ollama", "run", model_name]
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8'
            )
            out, err = proc.communicate(prompt)

            if proc.returncode != 0 and "memory" in err.lower():
                st.warning(f"Model `{model_name}` requires more memory than available. Falling back...")
                continue

            if proc.returncode != 0:
                st.error(f"Ollama Error with `{model_name}`: {err}")
                return None, model_name

            model_info_placeholder.success(f"Successfully using model: `{model_name}`")
            return out.strip(), model_name

        except FileNotFoundError:
            st.error("Ollama command not found. Ensure Ollama is installed and in your system's PATH.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred with Ollama: {e}")
            return None, model_name
    
    st.error("All attempted models failed. Please check your Ollama setup and available memory.")
    return None, None


def _build_prompt(question_id, scheme_points, student_answer):
    """Builds a highly explicit structured prompt to prevent hallucination."""
    scheme_text = "\n".join([f"- {pt['text']}" for pt in scheme_points])
    prompt = f"""
<|SYSTEM|>
You are an AI exam evaluator. Your task is to grade the student's answer based *only* on the provided marking scheme.
- Compare the student's answer *only* against the text in the 'Marking Scheme' section.
- Do not use any external knowledge.
- Your entire response must be a single, valid JSON object.
<|USER|>
**Marking Scheme for Question {question_id}:**
---
{scheme_text}
---

**Student's Answer:**
---
{student_answer}
---

**Instructions:**
Return a single JSON object with these exact keys:
- "marks_awarded": (integer) The total marks awarded based on the scheme.
- "max_marks": (integer) The total possible marks for this question.
- "confidence_score": (float, 0.0 to 1.0) Your confidence in the score.
- "justification": (string) A brief reason for your scoring, referencing the scheme.

Your response must start with {{ and end with }}.
"""
    return prompt.strip()

def _robust_json_parser(llm_output):
    """A highly resilient parser for LLM JSON output."""
    cleaned_output = re.sub(r'```json\s*|\s*```', '', llm_output).strip()
    typos = {'"marks_awaired"': '"marks_awarded"', '"confidence_scor"': '"confidence_score"'}
    for wrong, right in typos.items():
        cleaned_output = cleaned_output.replace(wrong, right)

    json_matches = re.findall(r'\{.*?\}', cleaned_output, re.DOTALL)
    if not json_matches:
        return {"error": "Parsing Failed", "details": "No JSON object found", "raw_output": llm_output}

    for match in json_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "marks_awarded" in data and "justification" in data:
                return data
        except json.JSONDecodeError:
            continue
    return {"error": "Parsing Failed", "details": "Found JSON-like objects, but none contained the required keys.", "raw_output": llm_output}

# --- Main Scoring Function ---
def score_answers(student_csv_path: Path, scheme_artifacts_dir: Path, output_csv_path: Path, model_info_placeholder):
    """Scores student answers using a direct lookup and an LLM with fallback."""
    
    ollama_model_list = ["mistral", "gemma:2b", "tinyllama"]

    meta = _load_meta(scheme_artifacts_dir / "scheme_meta.jsonl")
    df = pd.read_csv(student_csv_path, dtype=str).fillna("")

    q_to_points = {qid: [] for qid in df['question_id'].unique()}
    for point in meta:
        qid = str(point["question_id"])
        if qid in q_to_points:
            q_to_points[qid].append(point)

    results = []
    progress_bar = st.progress(0, text="Scoring answers...")

    for i, row in enumerate(df.itertuples()):
        qid = str(row.question_id)
        ans_text = row.answer_text

        scheme_points = q_to_points.get(qid, [])
        max_marks = sum(p.get('marks', 1) for p in scheme_points)

        res_json = {}
        if not scheme_points:
            res_json = {"marks_awarded": "N/A", "confidence_score": 0.0, "justification": "No marking scheme points found for this question ID."}
        else:
            prompt = _build_prompt(qid, scheme_points, ans_text)
            llm_output, used_model = _ollama_chat_with_fallback(ollama_model_list, model_info_placeholder, prompt)
            
            if llm_output:
                res_json = _robust_json_parser(llm_output)
                if "error" in res_json:
                    res_json['marks_awarded'] = "Error"
                    res_json['justification'] = f"Failed to parse LLM output: {res_json.get('raw_output', '')}"
            else:
                res_json = {"marks_awarded": "Error", "justification": f"Ollama model ({used_model}) did not return a response."}

        results.append({
            "student_id": row.student_id,
            "question_id": qid,
            "marks_awarded": res_json.get("marks_awarded"),
            "max_marks": max_marks,
            "confidence_score": res_json.get("confidence_score", 0.0),
            "justification": res_json.get("justification")
        })
        
        progress_bar.progress((i + 1) / len(df), text=f"Scoring Question {qid}...")

    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    progress_bar.empty()
