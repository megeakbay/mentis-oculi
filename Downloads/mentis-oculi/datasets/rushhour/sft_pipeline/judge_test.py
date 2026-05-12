"""
Quick sync test of the judge on 3 steps. No batch, no writes to cot_reasoning.json.
"""
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
import PIL.Image

from judge_submit import JUDGE_PROMPT, MODEL, _load_dotenv


SAMPLES = [
    ("level_01", "puzzle_0001", 0),
    ("level_03", "puzzle_0025", 0),
    ("level_05", "puzzle_0050", 4),
]


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output_sft_50").resolve()
    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for level, puzzle, step_idx in SAMPLES:
        pdir = output_dir / level / puzzle
        cot = json.loads((pdir / "cot_reasoning.json").read_text())
        step = next(s for s in cot if s.get("step") == step_idx)
        cur = pdir / ("initial.png" if step_idx == 0 else f"cot_{step_idx - 1:02d}.png")
        nxt = pdir / f"cot_{step_idx:02d}.png"

        prompt = JUDGE_PROMPT.replace("<<<REASONING>>>", step["reasoning"]).replace("<<<RESPONSE>>>", step["response"])
        print(f"\n=== {level}/{puzzle} step {step_idx} ===")
        print(f"  student reasoning: {step['reasoning'][:150]}...")
        print(f"  student response:  {step['response']}")
        resp = client.models.generate_content(
            model=MODEL,
            contents=[prompt, PIL.Image.open(cur), PIL.Image.open(nxt)],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        print(f"  --> judge raw: {resp.text}")
        try:
            data = json.loads(resp.text)
            if isinstance(data, list) and data:
                data = data[0]
            print(f"  parsed: is_correct_reasoning={data.get('is_correct_reasoning')} "
                  f"is_correct_no_hints={data.get('is_correct_no_hints')} "
                  f"reasoning={data.get('reasoning')!r}")
        except Exception as e:
            print(f"  PARSE FAIL: {e}")


if __name__ == "__main__":
    main()
