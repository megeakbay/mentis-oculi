import os
import json
from google import genai
from google.genai import types
from pathlib import Path
import PIL.Image


def _load_dotenv():
    """Minimal .env loader — walks up from this file looking for a .env."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


_load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)


STATIC_PROMPT = """
You are provided with four images of a form-board assembly puzzle.
1. The first image is the piece legend, showing the target silhouette and all five candidate pieces labeled A, B, C, D, E.
2. The second image is the current assembly state (pieces placed so far inside the silhouette; may be empty).
3. The third image is the next assembly state, acting as a visual hint for which single piece was just placed.
4. The fourth image is the fully-solved puzzle, showing where every solution piece ends up — use this to make your look-ahead reasoning grounded in the actual final layout.

Your task:
Compare the second and third images to identify which single piece (A, B, C, D, or E) was added in this step. Then write reasoning that justifies the choice GLOBALLY, not just locally.

Your reasoning MUST cover:
1. Why the chosen piece fits the open region right now (edges/angles/size cues from the legend and current state).
2. Look-ahead: what shape/region will remain after this placement, and why that remainder looks compatible with the pieces still available. Skip only if this placement completes the silhouette.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this move PURELY from the legend and the current state. You must NEVER mention the third image, the fourth image, the "next state", the "solved puzzle", or any "hint".
- Use the piece's letter label (A/B/C/D/E) to refer to it.
- Keep the reasoning to 2–5 short sentences. No informal ･･ blocks, no pauses, no roleplay.

Respond EXACTLY with this JSON format:
{
  "reasoning": "[Brief logical explanation of why this piece fits the remaining open region.]",
  "piece": "<LABEL>"
}
where <LABEL> is exactly one of A, B, C, D, E.
"""


def parse_step_response(text: str) -> dict:
    """Parse the JSON response. Returns {'piece': str|None, 'reasoning': str|None}."""
    text = text.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"piece": None, "reasoning": None}
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return {"piece": None, "reasoning": None}
    piece = obj.get("piece")
    if isinstance(piece, str):
        piece = piece.strip().upper()
        if piece not in {"A", "B", "C", "D", "E"}:
            piece = None
    else:
        piece = None
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = None
    return {"piece": piece, "reasoning": reasoning}


def generate_step_reasoning(legend_path, current_path, next_path, final_path):
    """
    Pass the static prompt and four images to Gemini and return (parsed_dict, raw_text).

    legend_path: path to combined.png (silhouette + all 5 labeled pieces)
    current_path: path to the previous state image (silhouette.png for step 0, else cot_{step-1:02d}.png)
    next_path: path to the next state image (cot_{step:02d}.png)
    final_path: path to bordered.png (fully-assembled puzzle)
    """
    contents = [STATIC_PROMPT]
    for path in [legend_path, current_path, next_path, final_path]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        contents.append(PIL.Image.open(p))

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    raw = response.text
    return parse_step_response(raw), raw
