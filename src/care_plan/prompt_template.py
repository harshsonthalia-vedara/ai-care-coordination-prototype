def build_care_plan_prompt(patient_data: dict, transcript: str) -> str:
    import json

    patient_json = json.dumps(patient_data, indent=2)

    return f"""
You are assisting a healthcare care coordinator.

Generate a structured Care Plan grounded ONLY in the provided patient data and transcript.

Requirements:
- Be clinically precise and data-driven.
- Explicitly reference lab values, symptoms, and adherence issues when relevant.
- Do NOT invent any information.
- Prioritize interventions based on urgency (High / Moderate / Low).
- Tie each intervention directly to a specific care gap.
- Include social determinants of health when present.

Output format:

Patient Summary (short paragraph)

Identified Care Gaps (bullet list)

Recommended Interventions:
- High Priority
- Moderate Priority
- Low Priority

Follow-Up Actions (bullet list)

Patient Data:
{patient_json}

Transcript:
{transcript}
"""