import argparse
import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List


def load_pickle_gz(filepath: str | Path) -> Any:
    """Load an object from a .pkl.gz file."""
    filepath = Path(filepath)

    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)


def extract_step_number(filename: str) -> int:
    """Extract the step number from a filename like step_12.pkl.gz."""
    match = re.search(r"step_(\d+)\.pkl\.gz$", filename)
    return int(match.group(1)) if match else -1


def is_valid_task_dir(path: Path) -> bool:
    """
    Check whether a path should be treated as a task directory.

    Rules:
    - Must be a directory.
    - Must not start with "_".
    """
    return path.is_dir() and not path.name.startswith("_")


def is_step_file(path: Path) -> bool:
    """
    Check whether a file is a valid step file.

    Rules:
    - Must be a file.
    - Filename must start with "step".
    - Filename must end with ".pkl.gz".
    """
    return (
        path.is_file()
        and path.name.startswith("step")
        and path.name.endswith(".pkl.gz")
    )


def get_sorted_step_files(task_dir: Path) -> List[Path]:
    """Return all step*.pkl.gz files sorted by step number."""
    step_files = [path for path in task_dir.iterdir() if is_step_file(path)]
    return sorted(step_files, key=lambda path: extract_step_number(path.name))


def parse_think_records(think: str) -> List[Dict[str, Any]]:
    """
    Parse the JSONL-style think field.

    Each line in think should be a JSON object.
    Invalid or empty lines are skipped and saved as ParseError records.
    """
    records = []

    for line in think.strip().splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(
                {
                    "agent_name": "ParseError",
                    "raw_record": line,
                }
            )

    return records


def format_record_for_txt(record: Dict[str, Any]) -> str:
    """Format one agent record into a human-readable text block."""
    agent_name = record.get("agent_name", "")
    input_text = record.get("input", "")
    output_text = record.get("output", "")
    reasoning_content = record.get("reasoning_content", "")

    return f"""
*** agent name: {agent_name}

*** input:
{input_text}

*** output:
{output_text}

*** reasoning content:
{reasoning_content}

"""


def extract_records_from_step_file(step_file: Path) -> List[Dict[str, Any]]:
    """Extract agent records from one step*.pkl.gz trajectory file."""
    trajectory = load_pickle_gz(step_file)
    agent_info = getattr(trajectory, "agent_info", None)

    if agent_info is None:
        return []

    think = getattr(agent_info, "think", None)

    if not think:
        return []

    return parse_think_records(think)


def write_message_record_json(task_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Write all extracted records to message_record.json."""
    output_path = task_dir / "message_record.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_message_record_txt(
    task_dir: Path,
    step_records: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Write all extracted records to message_record.txt in a human-readable form."""
    output_path = task_dir / "message_record.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for step_name, records in step_records.items():
            f.write(f"----------------------{step_name}-----------------------------\n")

            for record in records:
                f.write(format_record_for_txt(record))
                f.write("\n")


def process_one_task_dir(task_dir: Path) -> None:
    """
    Process one task directory and generate:
    - message_record.json
    - message_record.txt
    """
    all_records = []
    step_records = {}

    step_files = get_sorted_step_files(task_dir)

    for step_file in step_files:
        try:
            records = extract_records_from_step_file(step_file)
        except Exception as e:
            print(f"Error processing step file {step_file}: {e}")
            continue

        if not records:
            continue

        step_name = step_file.name.split(".")[0]
        step_records[step_name] = records
        all_records.extend(records)

    write_message_record_json(task_dir, all_records)
    write_message_record_txt(task_dir, step_records)

    print(f"Saved records for: {task_dir}")


def process_experiment_dir(experiment_dir: str | Path) -> None:
    """
    Process all valid task directories under an experiment directory.

    A valid task directory:
    - is a directory
    - does not start with "_"

    A valid step file:
    - starts with "step"
    - ends with ".pkl.gz"
    """
    experiment_dir = Path(experiment_dir)

    for task_dir in experiment_dir.iterdir():
        if not is_valid_task_dir(task_dir):
            continue

        try:
            process_one_task_dir(task_dir)
        except Exception as e:
            print(f"Error processing task directory {task_dir}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract readable message records from AgentLab trajectories."
    )
    parser.add_argument(
        "--record_dir",
        type=Path,
        required=True,
        help="Experiment directory containing per-task trajectory directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_experiment_dir(args.record_dir)


if __name__ == "__main__":
    main()
