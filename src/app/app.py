import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

PATIENT_PATH = DATA_DIR / "patient_case_01.json"
TRANSCRIPT_PATH = DATA_DIR / "transcript_case_01.txt"
GENERATED_PLAN_PATH = OUTPUT_DIR / "generated_care_plan_case_01.md"
APPROVED_PLAN_PATH = OUTPUT_DIR / "approved_care_plan_case_01.md"


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in Streamlit Secrets.")
    return OpenAI(api_key=api_key)


def load_patient_data():
    with open(PATIENT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript():
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_care_plan_prompt(patient_data: dict, transcript: str) -> str:
    patient_json = json.dumps(patient_data, indent=2)

    return f"""
You are assisting a healthcare care coordinator for a U.S. health payer.

Generate a structured Care Plan grounded ONLY in the provided patient data and transcript.

Requirements:
- Be clinically precise and data-driven.
- Explicitly reference lab values, symptoms, and adherence issues when relevant.
- Do NOT invent any information.
- Prioritize interventions based on urgency (High / Moderate / Low).
- Tie each intervention directly to a specific care gap.
- Include social determinants of health when present.
- Keep the tone suitable for a care coordinator review workflow.

Output format:

Patient Summary

Identified Care Gaps

Recommended Interventions
- High Priority
- Moderate Priority
- Low Priority

Follow-Up Actions

Patient Data:
{patient_json}

Transcript:
{transcript}
"""


def generate_care_plan(patient_data: dict, transcript: str) -> str:
    client = get_openai_client()
    prompt = build_care_plan_prompt(patient_data, transcript)

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )
    return response.output_text


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# UI

st.set_page_config(page_title="AI Care Coordination Prototype", layout="wide")

st.title("AI Care Coordination Prototype")
st.caption("Synthetic demo for AI-assisted care plan generation")

patient_data = load_patient_data()
transcript = load_transcript()

if "generated_plan" not in st.session_state:
    if GENERATED_PLAN_PATH.exists():
        st.session_state.generated_plan = GENERATED_PLAN_PATH.read_text(encoding="utf-8")
    else:
        st.session_state.generated_plan = ""

if "editable_plan" not in st.session_state:
    st.session_state.editable_plan = st.session_state.generated_plan

tab1, tab2, tab3 = st.tabs(["Patient Record", "Transcript", "Care Plan Review"])

with tab1:
    st.subheader("Patient Record")
    st.json(patient_data)

with tab2:
    st.subheader("Coordinator Call Transcript")
    st.text_area("Transcript", transcript, height=350, disabled=True)

with tab3:
    st.subheader("Draft Care Plan")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Draft Care Plan", use_container_width=True):
            with st.spinner("Generating care plan..."):
                try:
                    plan = generate_care_plan(patient_data, transcript)
                    st.session_state.generated_plan = plan
                    st.session_state.editable_plan = plan
                    save_text(GENERATED_PLAN_PATH, plan)
                    st.success("Draft care plan generated.")
                except Exception as e:
                    st.error(f"Error generating care plan: {e}")

    with col2:
        if st.button("Reload Saved Draft", use_container_width=True):
            if GENERATED_PLAN_PATH.exists():
                saved = GENERATED_PLAN_PATH.read_text(encoding="utf-8")
                st.session_state.generated_plan = saved
                st.session_state.editable_plan = saved
                st.success("Saved draft reloaded.")
            else:
                st.warning("No saved draft found yet.")

    st.markdown("### Editable Care Plan")
    edited_plan = st.text_area(
        "Coordinator Review",
        value=st.session_state.editable_plan,
        height=500
    )
    st.session_state.editable_plan = edited_plan

    col3, col4 = st.columns([1, 1])

    with col3:
        if st.button("Save Edited Draft", use_container_width=True):
            save_text(GENERATED_PLAN_PATH, st.session_state.editable_plan)
            st.success(f"Edited draft saved to {GENERATED_PLAN_PATH.name}")

    with col4:
        if st.button("Approve Final Care Plan", use_container_width=True):
            save_text(APPROVED_PLAN_PATH, st.session_state.editable_plan)
            st.success(f"Approved care plan saved to {APPROVED_PLAN_PATH.name}")

st.markdown("---")
st.markdown("### Demo Notes")
st.write(
    """
This prototype demonstrates a care coordination workflow in which structured patient data and a coordinator call transcript
are used to generate a draft care plan for human review, editing, and approval.
"""
)