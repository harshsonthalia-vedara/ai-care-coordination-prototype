# Prompt Strategy

## Objective
Generate a draft care plan grounded strictly in structured patient data and the coordinator call transcript.

## Approach
The prompt was designed to:
- use only the provided patient record and transcript
- avoid unsupported facts or hallucinated details
- produce a structured output suitable for coordinator review
- prioritize actions by urgency
- include both clinical and social barriers when present

## Guardrails
The model is instructed to:
- not invent diagnoses, labs, medications, or history
- explicitly reference available facts such as symptoms, HbA1c, adherence issues, and barriers
- keep the tone concise, clinical, and actionable

## Human-in-the-Loop
The generated care plan is treated as a draft only. A coordinator reviews, edits, and approves the final plan before use.