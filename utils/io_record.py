from dataclasses import dataclass, asdict
from typing import Optional, Dict
import time
import json
@dataclass
class AgentIO:
    agent_name: str
    task: str
    subtask: str
    axtree: str
    input: str
    output: str
    extracted_part: str
    reasoning_content: str
    timestamp: float
    step: int
    time_cost: Optional[float] = None
    metadata: Optional[Dict] = None

@dataclass
class AgentIO_sampling:
    agent_name: str
    task: str
    subtask: str
    subenv: str
    axtree: str
    input: str
    output: str
    extracted_part: str
    reasoning_content: str
    timestamp: float
    try_num: int
    step: int
    time_cost: Optional[float] = None
    metadata: Optional[Dict] = None


class InteractionLogger:
    def __init__(self):
        self.records = []

    def log(self, agent_io: AgentIO):
        self.records.append(agent_io)

    def to_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def print_summary(self):
        for r in self.records:
            print(f"[{r.step_id}] [{r.agent_name}] → {r.output[:60]}...")

    def save(self, agent_io: AgentIO, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(agent_io), ensure_ascii=False))
