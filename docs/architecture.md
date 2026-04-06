# Architecture Overview

## Purpose
This prototype demonstrates an AI-assisted care coordination workflow that generates a draft care plan from structured patient data and a simulated coordinator call transcript, with human review before final approval.

## Components
- Patient Data Layer: `data/patient_case_01.json`
- Transcript Input: `data/transcript_case_01.txt`
- Prompt / Orchestration Layer: `src/care_plan/`
- LLM Generation Layer: OpenAI API
- Coordinator Review Interface: `src/app/app.py`
- Outputs: `outputs/generated_care_plan_case_01.md`, `outputs/approved_care_plan_case_01.md`

## Workflow
1. Load structured patient record
2. Load coordinator transcript
3. Construct grounded prompt
4. Generate draft care plan using LLM
5. Present draft to coordinator in Streamlit interface
6. Allow manual editing and approval
7. Save generated and approved outputs

## Design Rationale
- Structured patient data provides longitudinal context
- Transcript provides real-time conversational context
- LLM is used for synthesis, not autonomous decision-making
- Human review is required before approval

## Production Considerations
- PHI-safe hosting and encryption
- Role-based access control
- Audit logging
- Prompt grounding to reduce hallucination risk
- Human-in-the-loop approval before final use

## Microsoft Stack Mapping
Although this prototype was implemented in Python and Streamlit, it maps closely to the intended Microsoft architecture:

- `patient_case_01.json` ↔ Dataverse / SharePoint
- OpenAI care plan generation ↔ AI Builder / Azure OpenAI
- Streamlit review interface ↔ Power Apps / Copilot Studio
- Python orchestration ↔ Power Automate