"""
Sanity-check that the judge catches a fabricated bad example.
Does NOT touch any cot_reasoning.json — purely in-memory test.
"""
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
import PIL.Image

from judge_submit import JUDGE_PROMPT, MODEL, _load_dotenv


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output_sft_50").resolve()
    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Real images from an existing puzzle, but synthetic bad student text.
    pdir = output_dir / "level_01" / "puzzle_0001"
    cur = pdir / "initial.png"
    nxt = pdir / "cot_00.png"

    # Bad: wrong action AND leaks the hint explicitly.
    bad_reasoning = (
        "Looking at the second image as a hint, I can see that rectangle A must move. "
        "In the first image, A is in the way of the red car reaching the exit, so A has to go up."
    )
    bad_response = "I decide to move rectangle A up."

    prompt = JUDGE_PROMPT.replace("<<<REASONING>>>", bad_reasoning).replace("<<<RESPONSE>>>", bad_response)
    print("=== Fabricated bad example (level_01/puzzle_0001 step 0 images) ===")
    print(f"  bad reasoning: {bad_reasoning}")
    print(f"  bad response:  {bad_response}")

    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt, PIL.Image.open(cur), PIL.Image.open(nxt)],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    print(f"\n  judge raw: {resp.text}")
    data = json.loads(resp.text)
    if isinstance(data, list) and data:
        data = data[0]
    print(f"  parsed: is_correct_reasoning={data.get('is_correct_reasoning')} "
          f"is_correct_no_hints={data.get('is_correct_no_hints')}")
    print(f"  judge reasoning: {data.get('reasoning')}")


if __name__ == "__main__":
    main()
