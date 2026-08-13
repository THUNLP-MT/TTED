import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_env_value(value, env_dict):
    """
    Replace placeholders like {BASE_URL} with values from env_dict.


    """
    if not isinstance(value, str):
        return value

    resolved = value

    for key, key_value in env_dict.items():
        placeholder = "{" + key + "}"
        if placeholder in resolved:
            resolved = resolved.replace(placeholder, str(key_value))

    return resolved


def register_environment(config):
    """
    Register all environment variables from config["env"].

    This function does not rename, merge, or remove any user-defined variable.
    """
    env_config = config.get("env", {})

    for key, value in env_config.items():
        resolved_value = resolve_env_value(value, env_config)
        os.environ[key] = str(resolved_value)


def load_agent_class(agent_config):
    module_name = agent_config["module"]
    class_name = agent_config["class_name"]

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_agent_kwargs(agent_config):
    """Build validated keyword arguments for the configured AgentArgs class."""
    if "temperature" not in agent_config:
        return {}

    temperature = agent_config["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError('config["agent"]["temperature"] must be a number.')

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('config["agent"]["temperature"] must be a finite, non-negative number.')

    return {"temperature": temperature}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a WebArena AgentLab study.")

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config_webarena.json")),
        help="Path to the WebArena config JSON file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("env", {})["PROJECT_ROOT"] = str(PROJECT_ROOT)

    register_environment(config)

    # These imports must happen after environment registration.
    from agentlab.experiments.study import make_study
    from utils.io_record import InteractionLogger

    agent_config = config["agent"]
    AgentArgsClass = load_agent_class(agent_config)

    logger = InteractionLogger()
    agent = AgentArgsClass(logger=logger, **load_agent_kwargs(agent_config))

    study_config = config.get("study", {})

    study = make_study(
        benchmark=study_config.get("benchmark", "webarena"),
        agent_args=[agent],
        comment=study_config.get("comment", "AgentLab study"),
    )

    study.run(n_jobs=int(study_config.get("n_jobs", 1)))


if __name__ == "__main__":
    main()
