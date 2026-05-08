import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = {
    1: ROOT / "01_langsmith_rag_pipeline.py",
    2: ROOT / "02_prompt_hub_ab_routing.py",
    3: ROOT / "03_ragas_evaluation.py",
    4: ROOT / "04_guardrails_validator.py",
}


def run_step(step: int) -> int:
    script = STEPS[step]
    print(f"\n{'=' * 72}\nRunning step {step}: {script.name}\n{'=' * 72}")
    completed = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 22 lab steps.")
    parser.add_argument("--step", type=int, choices=sorted(STEPS), help="Run only one step")
    args = parser.parse_args()

    if args.step:
        raise SystemExit(run_step(args.step))

    for step in sorted(STEPS):
        code = run_step(step)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
