import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt_template import build_care_plan_prompt


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)

    base_dir = Path(__file__).resolve().parents[2]
    patient_path = base_dir / "data" / "patient_case_01.json"
    transcript_path = base_dir / "data" / "transcript_case_01.txt"
    output_path = base_dir / "outputs" / "generated_care_plan_case_01.md"

    with open(patient_path, "r", encoding="utf-8") as f:
        patient_data = json.load(f)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    prompt = build_care_plan_prompt(patient_data, transcript)

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
    )

    care_plan = response.output_text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(care_plan)

    print("\nGenerated Care Plan:\n")
    print(care_plan)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()