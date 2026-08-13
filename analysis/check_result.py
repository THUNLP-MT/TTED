import argparse
import json
from pathlib import Path
from typing import Iterable


def extract_task_name(directory_name: str) -> str:
    """Extract an AgentLab task id such as ``webarena.763``."""
    prefix, separator, retry = directory_name.rpartition("_")
    if separator and retry.isdigit():
        candidate = prefix.rsplit("_", 1)[-1]
        if "." in candidate:
            return candidate
    return directory_name


def iter_task_dirs(root_dir: str | Path) -> Iterable[Path]:
    root_path = Path(root_dir)
    for path in sorted(root_path.iterdir()):
        if path.is_dir() and not path.name.startswith("_"):
            yield path


def load_reward(task_dir: Path):
    summary_path = task_dir / "summary_info.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("cum_reward")


def load_result(root_dir: str | Path):
    success_id = []
    un_success_id = []

    for task_dir in iter_task_dirs(root_dir):
        task_name = extract_task_name(task_dir.name)
        reward = load_reward(task_dir)

        if reward is not None and int(reward) == 1:
            success_id.append(task_name)
        else:
            un_success_id.append(task_name)

    return success_id, un_success_id


def load_result_path(root_dir: str | Path):
    success_id = []
    un_success_id = []

    for task_dir in iter_task_dirs(root_dir):
        reward = load_reward(task_dir)
        task_path = str(task_dir)

        if reward is not None and int(reward) == 1:
            success_id.append(task_path)
        else:
            un_success_id.append(task_path)

    return success_id, un_success_id


def successful_check(root_dirs: Iterable[str | Path], task: str):
    success_id = []
    un_success_id = []
    fail_tasks = []

    for root_dir in root_dirs:
        for task_dir in iter_task_dirs(root_dir):
            if extract_task_name(task_dir.name) != task:
                continue

            reward = load_reward(task_dir)
            task_path = str(task_dir)
            if reward is None:
                fail_tasks.append(task_path)
            elif int(reward) == 1:
                success_id.append(task_path)
            else:
                un_success_id.append(task_path)

    return success_id, un_success_id, fail_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize successful and unsuccessful AgentLab task runs."
    )
    parser.add_argument(
        "--record_dir",
        type=Path,
        required=True,
        help="Experiment directory containing per-task trajectory directories.",
    )
    parser.add_argument(
        "--show_paths",
        action="store_true",
        help="Print task directory paths instead of task ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loader = load_result_path if args.show_paths else load_result
    success, unsuccessful = loader(args.record_dir)
    print("success tasks:", success, len(success))
    print("unsuccessful tasks:", unsuccessful, len(unsuccessful))


if __name__ == "__main__":
    main()
