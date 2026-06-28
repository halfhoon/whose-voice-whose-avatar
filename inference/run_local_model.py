from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference.config import LOCAL_MODELS, TASK_TO_CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local/open-weight models for Task 1, Task 2, or all released conditions.")
    parser.add_argument("--model", required=True, choices=LOCAL_MODELS)
    parser.add_argument("--task", default="all", choices=TASK_TO_CONDITIONS.keys())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    from inference.runner import run_selected_task

    asyncio.run(
        run_selected_task(
            model_name=args.model,
            task_name=args.task,
            batch_size=args.batch_size,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
