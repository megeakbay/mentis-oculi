import os
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
You are provided with two images of a block sliding puzzle.
1. The first image is the current state.
2. The second image shows the exact next state, acting as a visual hint.

Each rectangle has a thin rail overlay with an arrow showing its local drive axis. A rectangle can only translate along that rail — either FORWARD (in the direction the arrow points) or BACKWARD (opposite the arrow). No rotations, no sideways motion. "forward" and "backward" are always relative to that specific rectangle's arrow, NOT relative to the page, compass, or viewer.

Your task:
Compare the two images to identify which single rectangle moved and in which direction. Then write reasoning for WHY that move helps at THIS step.

STRICT REASONING RULES:
1. Only reason about what is physically visible in the current state (first image). Do not invent constraints — if you claim a rectangle cannot move in some direction, it must be visually obvious from the first image that another rectangle or the board boundary is blocking it.
2. Only justify THIS move. Do not predict or describe future moves. Do not say what will happen after this move.
3. Do not claim a rectangle is blocked by an obstacle unless that obstacle is clearly overlapping or adjacent to its rail path in the first image.
4. Keep reasoning to 2-3 sentences maximum.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this move PURELY from looking at the first image. You must NEVER mention the second image, the "next state", or any "hint".
- Describe directions ONLY as "forward" or "backward" relative to each rectangle's own rail arrow. Do NOT use absolute directions like "up", "down", "left", "right".

Respond EXACTLY with this JSON format:
{
  "reasoning": "[2-3 sentences explaining why this move helps at this step, based only on what is visible.]",
  "response": "<LABEL> forward"
}
The response field is exactly two tokens: the rectangle label, then either 'forward' or 'backward'. No sentence, no punctuation.
"""


def generate_step_reasoning(current_image_path, next_image_path):
    """
    Passes the static prompt and two images to Gemini and returns the parsed JSON dict.
    """
    contents = [STATIC_PROMPT]

    for path in [current_image_path, next_image_path]:
        if Path(path).exists():
            contents.append(PIL.Image.open(path))
        else:
            print(f"Error: Could not find image at {path}")
            return None

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return response.text
