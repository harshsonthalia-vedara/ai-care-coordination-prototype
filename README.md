# AI Care Coordination Prototype

This project is a proof-of-concept AI-assisted care coordination workflow that generates a grounded draft care plan from structured patient data and a simulated care coordinator call transcript, followed by human review and approval.

The goal of this system is to reduce the administrative burden on care coordinators by shifting the workflow from manual documentation and synthesis to AI-assisted drafting and human validation.

## Live Demo

[Add your Streamlit app link here]
- Local URL: http://localhost:8501
- Network URL: http://192.168.86.37:8501

## Problem Statement

Care coordinators today must:
- manage conversations with patients
- navigate multiple systems
- manually synthesize patient history
- draft care plans after the call

This process is time-consuming and limits scalability.

### Objective
Generate a **draft care plan in real time during or immediately after the call**, allowing the coordinator to review and approve rather than create from scratch.

## Solution Overview

This prototype demonstrates an end-to-end workflow:

1. Structured patient data is loaded  
2. A coordinator-patient transcript is processed  
3. An LLM generates a **grounded draft care plan**  
4. The coordinator reviews, edits, and approves the plan  

## Features

- Synthetic patient data modeling
- Transcript-based care plan generation
- Grounded LLM output (no hallucinated data)
- Prioritized interventions (High / Moderate / Low)
- Human-in-the-loop review and approval
- Streamlit-based interactive UI
- Saved draft and approved outputs
- Optional sentiment + intent classification (dataset task)



## Project Structure

```bash
ai-care-coordination-prototype/
│
├── data/
│   ├── patient_case_01.json
│   ├── transcript_case_01.txt
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
│
├── src/
│   ├── care_plan/
│   ├── classifier/
│   └── app/
│       └── app.py
│
├── outputs/
│   ├── generated_care_plan_case_01.md
│   └── approved_care_plan_case_01.md
│
├── docs/
│   ├── schema.md
│   ├── architecture.md
│   └── prompt_strategy.md
│
├── requirements.txt
└── README.md

