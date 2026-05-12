"""
Verify the LLM's stated action per step matches the ground-truth action
recorded in metadata.json. Prints a mismatch list.
"""
import json
import re
import sys
from pathlib import Path


DIR_WORDS = {
    "right": 1, "left": -1, "up": 2, "down": -2,
}


def gt_label(object_id: str, objects: list) -> str:
    if object_id == "red_car":
        return "R"
    for o in objects:
        if o.get("id") == object_id:
            return o.get("label") or "?"
    return "?"


def gt_axis(object_id: str, objects: list):
    for o in objects:
        if o.get("id") == object_id:
            return o.get("local_axis")
    return None


def extract_label(text: str) -> str:
    t = text
    if re.search(r"\bred\b", t, re.I) or re.search(r"\bR\b", t):
        if re.search(r"\b(red (?:car|rectangle|block)|rectangle R|block R|car R|\bR\b)", t, re.I):
            return "R"
    # Any standalone single uppercase letter A-Z (excluding R handled above)
    m = re.findall(r"\b([A-Z])\b", t)
    for letter in m:
        if letter != "I":
            return letter
    return None


def extract_directions(text: str):
    tl = text.lower()
    found = set()
    for w, code in DIR_WORDS.items():
        if re.search(rf"\b{w}\b", tl):
            found.add(code)
    return found


def directions_match(pred_dirs: set, gt_dir: int, axis) -> bool:
    """
    gt_dir: +/-1 (horizontal) or +/-2 (vertical) — along local_axis.
    For axis-aligned cars, axis is ~ (1,0) or (0,1), so gt_dir maps cleanly.
    For rotated (diagonal) cars, both a horizontal and a vertical word can
    legitimately describe the motion. Accept if any predicted direction's
    component sign matches the axis-projected motion.
    """
    if not pred_dirs:
        return False
    ax_x, ax_y = axis if axis else (1.0, 0.0)
    sign = 1 if gt_dir > 0 else -1
    vx = ax_x * sign
    vy = ax_y * sign
    EPS = 0.2
    for code in pred_dirs:
        if code == 1 and vx > EPS: return True
        if code == -1 and vx < -EPS: return True
        if code == 2 and vy > EPS: return True
        if code == -2 and vy < -EPS: return True
    return False


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_sft_50")
    root = root.resolve()
    total = 0
    label_mismatch = 0
    dir_mismatch = 0
    unparseable = 0
    mismatches = []
    for meta_f in sorted(root.glob("level_*/puzzle_*/metadata.json")):
        meta = json.loads(meta_f.read_text())
        cot_f = meta_f.parent / "cot_reasoning.json"
        if not cot_f.exists(): continue
        steps = json.loads(cot_f.read_text())
        actions = meta.get("actions", [])
        objects = meta.get("objects", [])
        for step in steps:
            if not isinstance(step, dict): continue
            idx = step.get("step")
            if idx is None or idx >= len(actions): continue
            gt = actions[idx]
            resp = step.get("response") or ""
            total += 1
            pred_lbl = extract_label(resp)
            pred_dirs = extract_directions(resp)
            gt_lbl = gt_label(gt["object_id"], objects)
            axis = gt_axis(gt["object_id"], objects)
            if pred_lbl is None and not pred_dirs:
                unparseable += 1
                mismatches.append((meta_f.parent, idx, "UNPARSEABLE", gt_lbl, gt["direction"], resp))
                continue
            lbl_ok = (pred_lbl == gt_lbl)
            dir_ok = directions_match(pred_dirs, gt["direction"], axis)
            if not lbl_ok: label_mismatch += 1
            if not dir_ok: dir_mismatch += 1
            if not (lbl_ok and dir_ok):
                reason = []
                if not lbl_ok: reason.append(f"label pred={pred_lbl} gt={gt_lbl}")
                if not dir_ok: reason.append(f"dir pred={pred_dirs} gt={gt['direction']} axis={axis}")
                mismatches.append((meta_f.parent, idx, "; ".join(reason), gt_lbl, gt["direction"], resp))
    print(f"total checked: {total}")
    print(f"label mismatch: {label_mismatch}")
    print(f"direction mismatch: {dir_mismatch}")
    print(f"unparseable: {unparseable}")
    print(f"mismatch rows: {len(mismatches)}")
    print()
    for m in mismatches:
        puzzle, idx, reason, gt_lbl, gt_dir, resp = m
        print(f"--- {puzzle.parent.name}/{puzzle.name} step {idx}")
        print(f"    {reason}")
        print(f"    gt: label={gt_lbl} direction={gt_dir}")
        print(f"    response: {resp}")


if __name__ == "__main__":
    main()
