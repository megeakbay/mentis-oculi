"""
Use Gemini to generate a semantic description of each puzzle's target image,
then store it in the puzzle's metadata.json as "target_semantic".

Run from the sliding-puzzle directory:
    python create_semantic.py --output-dir output/level_03
    python create_semantic.py --output-dir output  # processes all levels
"""

import argparse
import json
import os
import time
from pathlib import Path

import PIL.Image
from google import genai


def _load_dotenv() -> None:
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
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SEMANTIC_PROMPT = """Divide this image into four quadrants and write one short sentence per quadrant describing what is visually there. Keep each sentence brief and natural — like "blue sky with a sun in the corner" or "grass with a bit of sky at the top". Mention what is present and roughly where, but do not list colors mechanically.

Output exactly four lines:
Top-left: <one sentence>
Top-right: <one sentence>
Bottom-left: <one sentence>
Bottom-right: <one sentence>

Nothing else."""


def describe_target(image_path: Path) -> str:
    img = PIL.Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[SEMANTIC_PROMPT, img],
    )
    return response.text.strip()


def process_puzzle_dir(puzzle_dir: Path, overwrite: bool = False) -> bool:
    semantic_path = puzzle_dir / "semantic.json"
    target_path = puzzle_dir / "target.png"

    if not target_path.exists():
        return False

    if not overwrite and semantic_path.exists():
        print(f"  [skip] {puzzle_dir.name} already has semantic.json")
        return False

    print(f"  [gen]  {puzzle_dir.name} ...", end=" ", flush=True)
    description = describe_target(target_path)

    semantic = {
        "puzzle_id": puzzle_dir.name,
        "target_image": "target.png",
        "target_semantic": description,
    }

    with open(semantic_path, "w") as f:
        json.dump(semantic, f, indent=2)

    print("done")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add semantic descriptions to sliding puzzle targets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root output directory (searched recursively for puzzle_XXXX folders)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate descriptions even if they already exist",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    puzzle_dirs = sorted(root.rglob("puzzle_*/"))

    if not puzzle_dirs:
        print(f"No puzzle directories found under {root}")
        return

    print(f"Found {len(puzzle_dirs)} puzzle directories under {root}")
    updated = 0

    for puzzle_dir in puzzle_dirs:
        changed = process_puzzle_dir(puzzle_dir, overwrite=args.overwrite)
        if changed:
            updated += 1
            time.sleep(args.delay)

    print(f"\nDone. Updated {updated}/{len(puzzle_dirs)} puzzles.")


if __name__ == "__main__":
    main()
