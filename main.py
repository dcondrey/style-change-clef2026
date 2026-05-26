"""TIRA entry point for PAN-CLEF 2026 Style Change Detection (MAWSA26).

Usage (TIRA convention):
  python main.py -i /input -o /output

Input: directory of problem-*.txt files (multi-paragraph documents).
Output: per-problem JSON files with multi-author, changes, and id fields.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from solution_2031_paradigm import PANCLEF2026_WinningSolution


def main():
    parser = argparse.ArgumentParser(
        description="PAN-CLEF 2026 Style Change Detection (TIRA)"
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  PAN-CLEF 2026: Style Change Detection (MAWSA26)")
    log.info("=" * 60)

    system = PANCLEF2026_WinningSolution("HuggingFaceTB/SmolLM-135M")

    problem_files = sorted(input_dir.glob("problem-*.txt"))
    if not problem_files:
        problem_files = sorted(input_dir.rglob("problem-*.txt"))

    log.info("  Found %d problem files", len(problem_files))

    for i, pf in enumerate(problem_files):
        text = pf.read_text(encoding="utf-8").strip()
        prob_id = pf.stem

        change_vector = system.predict_style_changes(text)

        result = {"changes": change_vector}

        # Mirror input subdirectory structure in output (e.g., easy/, hard/)
        rel = pf.relative_to(input_dir)
        out_subdir = output_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file = out_subdir / f"solution-{prob_id}.json"
        with open(out_file, "w") as f:
            json.dump(result, f)

        if (i + 1) % 100 == 0:
            log.info("  Processed %d / %d", i + 1, len(problem_files))

    log.info("  Done: %d predictions written to %s", len(problem_files), output_dir)


if __name__ == "__main__":
    main()
