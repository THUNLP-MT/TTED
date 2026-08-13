from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import bgym
from agentlab.agents.agent_args import AgentArgs
from bgym import Agent
from openai import OpenAI
from utils.io_record import AgentIO, InteractionLogger
from utils.prompt_packer import truncate_prompt
from utils.subenv_expand import expand_subenv_once_to_text_wide_with_metrics



DEFAULT_LLM_API_KEY = os.getenv("EVAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
DEFAULT_LLM_API_BASE = os.getenv("EVAL_OPENAI_API_BASE", os.getenv("OPENAI_API_BASE"))
DEFAULT_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", os.getenv("AGENT_MODEL_NAME", ""))
DEFAULT_ERROR_LOG_DIR = os.getenv("AGENT_ERROR_LOG_DIR", "./logs/agent_errors")
DEFAULT_TOKENIZER_NAME = os.getenv("AGENT_TOKENIZER_NAME", "utils/tokenizer")
DEFAULT_ACTION_RETRY = int(os.getenv("AGENT_ACTION_RETRY", "5"))
DEFAULT_JSON_RETRY = int(os.getenv("AGENT_JSON_RETRY", "5"))
DEFAULT_SAMPLING_NUM = int(os.getenv("AGENT_SAMPLING_NUM", "4"))

NO_HISTORY_MESSAGE = "No history, this is your first step."
NO_PARENT_HISTORY_MESSAGE = "No history, this is the first step."
NO_FAILURE_MESSAGE = "This is your first attempt."

ACTION_RESPONSE_PATTERN = re.compile(
    r"### Manipulate element:\s*(.*?)\s*### Action:\s*(.*?)\s*### Action intent:\s*(.*)",
    flags=re.DOTALL,
)
CODE_BLOCK_PATTERN = re.compile(r"```(.*?)```", flags=re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*?\}")


@dataclass
class HistoryNode:
    """A node in the interaction history tree."""

    parent: Optional["HistoryNode"] = field(default=None, repr=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: Optional[str] = None
    element: Optional[str] = None
    intent: Optional[str] = None
    url: Optional[str] = None
    axtree: Optional[str] = None
    subtask: Optional[str] = None
    subenv: Optional[str] = None
    is_error: bool = False
    error_info: Optional[str] = None
    children: list["HistoryNode"] = field(default_factory=list, repr=False)

    def add_child(self, child: "HistoryNode") -> None:
        self.children.append(child)

    def mark_as_error(self, error_info: Optional[str] = None) -> None:
        self.is_error = True
        self.error_info = error_info

    def __repr__(self) -> str:
        return f"HistoryNode(id={self.id[:6]}, action={self.action}, url={self.url})"


class HistoryTree:
    """Store accepted actions and failed alternative attempts."""

    def __init__(self) -> None:
        self.root = HistoryNode()
        self.current = self.root
        self.nodes: dict[str, HistoryNode] = {self.root.id: self.root}

    def add_empty_node(self) -> HistoryNode:
        node = HistoryNode(parent=self.current)
        self.current.add_child(node)
        self.current = node
        self.nodes[node.id] = node
        return node

    def backtrack(self) -> None:
        if self.current.parent is None:
            raise RuntimeError("Cannot backtrack from the root node.")
        self.current = self.current.parent

    def get_current_node(self) -> HistoryNode:
        return self.current

    def get_history(self) -> str:
        return self._format_history_until(self.current, empty_message=NO_HISTORY_MESSAGE)

    def get_n_history(self, n: int) -> str:
        history = self._history_rows_until(self.current)
        if not history:
            return NO_HISTORY_MESSAGE
        return "\n".join(history[-n:])

    def parent_history(self) -> str:
        parent = self.current.parent
        if parent is None or parent.action is None:
            return NO_PARENT_HISTORY_MESSAGE
        return self._format_history_until(parent, empty_message=NO_PARENT_HISTORY_MESSAGE)

    def get_error_descendants(self) -> str:
        errors = [child.error_info for child in self.current.children if child.is_error and child.error_info]
        return "\n".join(errors) if errors else NO_FAILURE_MESSAGE

    def _format_history_until(self, node: Optional[HistoryNode], empty_message: str) -> str:
        rows = self._history_rows_until(node)
        return "\n".join(rows) if rows else empty_message

    @staticmethod
    def _history_rows_until(node: Optional[HistoryNode]) -> list[str]:
        raw_rows: list[str] = []
        while node is not None and node.action is not None:
            raw_rows.append(
                "url: {url}, action: {action}, action summary: {summary}".format(
                    url=node.url,
                    action=node.action,
                    summary=node.intent,
                )
            )
            node = node.parent

        rows: list[str] = []
        for idx, row in enumerate(reversed(raw_rows), start=1):
            rows.append(f"Step {idx}. {row}")
        return rows


class CustomAgent(Agent):
    """Action + decomposition + summary agent.

    The agent receives environment information exactly through the normal
    AgentLab observation dictionary: ``obs['axtree_txt']``, ``obs['url']``,
    and ``obs['goal']``. The full AXTree is first used by PlanningAgent and
    EnvSegAgent. The extracted sub-environment is then used only inside the
    action prompt, while logging still keeps the original full AXTree.
    """

    def __init__(
        self,
        temperature: float,
        logger: Optional["InteractionLogger"],
        llm_api_base: str = DEFAULT_LLM_API_BASE,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        llm_model_name: str = DEFAULT_MODEL_NAME,
        error_log_dir: str = DEFAULT_ERROR_LOG_DIR,
    ) -> None:
        self.temperature = temperature
        self.action_set = bgym.HighLevelActionSet(["webarena"], multiaction=False)
        self.counter = 0
        self.history = HistoryTree()
        self.logger = logger
        self.error_log_dir = error_log_dir
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self._client: Optional[OpenAI] = None

    def get_action(self, obs: Any) -> tuple[str, bgym.AgentInfo]:
        """Plan a sub-task, extract a sub-environment, choose an action, and summarize it."""
        start_time = time.perf_counter()

        if self._is_blank_page(obs):
            print("blank page")
            return "go_forward()", bgym.AgentInfo()

        self._initialize_root_if_needed(obs)
        self.counter += 1

        task = obs["goal"]
        full_axtree = obs["axtree_txt"]
        url = obs["url"]
        think_records: list[dict[str, Any]] = []

        subtask, planning_record = self.task_decomp(task=task, axtree=full_axtree)
        think_records.append(planning_record)

        env_info, env_record, env_ok = self.env_decomp(subtask=subtask, axtree=full_axtree, task=task)
        think_records.append(env_record)
        action_environment = env_info[1] if env_ok else full_axtree

        action_candidates, sampling_records = self._sample_action_candidates(
            task=task,
            subtask=subtask,
            axtree=full_axtree,
            action_environment=action_environment,
            url=url,
        )
        think_records.extend(sampling_records)

        selected_idx = random.randrange(len(action_candidates))
        selected_candidate = action_candidates[selected_idx]
        action_info = selected_candidate["action_info"]
        action_prompt = selected_candidate["prompt"]
        response = selected_candidate["response"]
        reasoning_content = selected_candidate["reasoning_content"]
        action_time = selected_candidate["time_cost"]

        action = action_info["action"]
        print(action)

        action_record = self._save_record(
            agent_name="ActionAgent",
            task=task,
            subtask=subtask,
            axtree=full_axtree,
            input_text=action_prompt,
            output_text=response,
            extracted_part=action_info,
            reasoning_content=reasoning_content,
            step=self.counter,
            time_cost=action_time,
            try_num=selected_idx,
        )
        think_records.append(action_record)

        current_node = self._append_action_to_history(
            obs=obs,
            action_info=action_info,
            subtask=subtask,
            subenv=action_environment,
        )

        summary, summary_record = self.summary_agent(
            task=task,
            subtask=subtask,
            action=action,
            intent=action_info["action_intent"],
            axtree=full_axtree,
            element=action_info["manipulate_element"],
        )
        current_node.intent = summary
        think_records.append(summary_record)

        elapsed = time.perf_counter() - start_time
        think_records.append({"time": elapsed})

        return action, bgym.AgentInfo(think=self._format_think(think_records))

    def _initialize_root_if_needed(self, obs: Any) -> None:
        if self.counter != 0:
            return
        self.history.root.axtree = obs["axtree_txt"]
        self.history.root.url = obs["url"]

    @staticmethod
    def _is_blank_page(obs: Any) -> bool:
        return "RootWebArea ''" in obs.get("axtree_txt", "")

    def _append_action_to_history(
        self,
        obs: Any,
        action_info: dict[str, str],
        subtask: str,
        subenv: str,
    ) -> HistoryNode:
        self.history.add_empty_node()
        node = self.history.get_current_node()
        node.action = action_info["action"]
        node.element = action_info["manipulate_element"]
        node.intent = action_info["action_intent"]
        node.axtree = obs["axtree_txt"]
        node.url = obs["url"]
        node.subtask = subtask
        node.subenv = subenv
        return node

    def task_decomp(self, task: str, axtree: str) -> tuple[str, dict[str, Any]]:
        """Generate the next sub-task using the full current AXTree."""
        start_time = time.perf_counter()
        prompt = self._build_planning_prompt(task=task, axtree=axtree)
        prompt = self._truncate_prompt_if_needed(prompt=prompt, environment=axtree, history=self.history)

        reasoning_content, response = self.llm(prompt)
        planning_output = self._extract_planning_output(response=response, prompt=prompt)
        subtask = planning_output["sub_task"]

        record = self._save_record(
            agent_name="PlanningAgent",
            task=task,
            subtask=subtask,
            axtree=axtree,
            input_text=prompt,
            output_text=response,
            extracted_part=planning_output,
            reasoning_content=reasoning_content,
            step=self.counter,
            time_cost=time.perf_counter() - start_time,
        )
        return subtask, record

    def env_decomp(self, subtask: str, axtree: str, task: str) -> tuple[list[Any], dict[str, Any], bool]:
        """Extract a task-relevant sub-environment from the full AXTree."""
        start_time = time.perf_counter()
        prompt = self._build_env_prompt(subtask=subtask, axtree=axtree)
        prompt = self._truncate_prompt_if_needed(prompt=prompt, environment=axtree, history=None, max_tokens=30000)

        last_reasoning: Optional[str] = None
        last_response = ""
        html_blocks = axtree
        expanded_text = axtree
        metrics = None
        success = False

        for try_idx in range(DEFAULT_JSON_RETRY):
            try:
                last_reasoning, last_response = self.llm(prompt, thinking_budget=5000, output_budget=5000)
                extracted = self._extract_code_block(last_response)
                expanded_text, metrics = expand_subenv_once_to_text_wide_with_metrics(
                    original_text=axtree,
                    subenv_text=extracted,
                    mode="focused",
                )
                html_blocks = extracted
                success = True
                break
            except Exception as exc:
                print(f"(EnvSegAgent) Format or expansion error, try {try_idx + 1}/{DEFAULT_JSON_RETRY}: {exc}")
                self.save_with_timestamp(
                    content=(
                        "(EnvSegAgent) Format or expansion error.\n"
                        f"Error: {exc}\n\nInput:\n{prompt}\n\nReasoning:\n{last_reasoning}\n\nOutput:\n{last_response}"
                    ),
                    directory=self.error_log_dir,
                )

        env_info: list[Any] = [html_blocks, expanded_text, metrics]
        record = self._save_record(
            agent_name="EnvSegAgent",
            task=task,
            subtask=subtask,
            axtree=axtree,
            input_text=prompt,
            output_text=last_response,
            extracted_part=env_info,
            reasoning_content=last_reasoning,
            step=self.counter,
            time_cost=time.perf_counter() - start_time,
        )
        return env_info, record, success

    def _sample_action_candidates(
        self,
        task: str,
        subtask: str,
        axtree: str,
        action_environment: str,
        url: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate multiple action candidates and record each candidate as Sampling."""
        candidates: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        for sample_idx in range(DEFAULT_SAMPLING_NUM):
            reasoning_content, response, action_info, action_prompt, action_time = self._query_action(
                task=task,
                subtask=subtask,
                axtree=axtree,
                action_environment=action_environment,
                url=url,
            )
            candidates.append(
                {
                    "reasoning_content": reasoning_content,
                    "response": response,
                    "action_info": action_info,
                    "prompt": action_prompt,
                    "time_cost": action_time,
                    "try_num": sample_idx,
                }
            )
            records.append(
                self._save_record(
                    agent_name="Sampling",
                    task=task,
                    subtask=subtask,
                    axtree=axtree,
                    input_text=action_prompt,
                    output_text=response,
                    extracted_part=action_info,
                    reasoning_content=reasoning_content,
                    step=self.counter,
                    time_cost=action_time,
                    try_num=sample_idx,
                )
            )

        return candidates, records

    def _query_action(
        self,
        task: str,
        subtask: str,
        axtree: str,
        action_environment: str,
        url: str,
    ) -> tuple[Optional[str], str, dict[str, str], str, float]:
        """Generate one executable action from the extracted action environment."""
        start_time = time.perf_counter()
        prompt = self._build_action_prompt(
            task=task,
            subtask=subtask,
            action_environment=action_environment,
            url=url,
        )
        prompt = self._truncate_prompt_if_needed(prompt=prompt, environment=action_environment, history=self.history)

        last_reasoning: Optional[str] = None
        last_response = ""
        for _ in range(DEFAULT_ACTION_RETRY):
            try:
                last_reasoning, last_response = self.llm(prompt)
                parsed = self._parse_action_response(last_response)
                return last_reasoning, last_response, parsed, prompt, time.perf_counter() - start_time
            except Exception as exc:
                print(f"(ActionAgent) Format error: {exc}")

        self.save_with_timestamp(
            content=(
                "(ActionAgent) The response does not match the expected format.\n"
                f"Input:\n{prompt}\n\nOutput:\n{last_response}"
            ),
            directory=self.error_log_dir,
        )
        raise ValueError("ActionAgent output does not match the expected format.")

    def summary_agent(
        self,
        task: str,
        subtask: str,
        action: str,
        intent: str,
        axtree: str,
        element: str,
    ) -> tuple[str, dict[str, Any]]:
        """Summarize the selected action and the current page context."""
        start_time = time.perf_counter()
        prompt = self._build_summary_prompt(
            task=task,
            subtask=subtask,
            action=action,
            intent=intent,
            axtree=axtree,
            element=element,
        )
        prompt = self._truncate_prompt_if_needed(prompt=prompt, environment=axtree, history=None)

        result: Optional[dict[str, Any]] = None
        last_reasoning: Optional[str] = None
        last_response = ""
        for _ in range(DEFAULT_JSON_RETRY):
            last_reasoning, last_response = self.llm(prompt)
            result = self.extract_llm_json(agent_name="SummaryAgent", prompt=prompt, output_text=last_response)
            if result:
                break

        if not result:
            raise ValueError("Failed to extract JSON from SummaryAgent output.")

        summary_text = (
            f"environment_description: {result.get('environment_description', '')}, "
            f"manipulated_element: {result.get('manipulated_element', '')}, "
            f"action_summary: {result.get('action_summary', '')}"
        )
        record = self._save_record(
            agent_name="SummaryAgent",
            task=task,
            subtask=subtask,
            axtree=axtree,
            input_text=prompt,
            output_text=last_response,
            extracted_part=result,
            reasoning_content=last_reasoning,
            step=self.counter,
            time_cost=time.perf_counter() - start_time,
        )
        return summary_text, record

    def llm(
        self,
        prompt: str,
        thinking_budget: int = 5000,
        output_budget: int = 5000,
    ) -> tuple[Optional[str], str]:
        """Call an OpenAI-compatible endpoint."""
        if self._client is None:
            self._client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_api_base)

        model = self.llm_model_name or self._client.models.list().data[0].id
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=thinking_budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        content = message.content or ""

        if response.choices[0].finish_reason == "length":
            truncated_reasoning = (
                f"<think>\n{reasoning_content}\n</think>" if reasoning_content is not None else content + "\n</think>"
            )
            followup = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt + "\n" + truncated_reasoning}],
                temperature=self.temperature,
                max_tokens=output_budget,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return truncated_reasoning, followup.choices[0].message.content or ""

        return reasoning_content, content

    def _build_planning_prompt(self, task: str, axtree: str) -> str:
        return f"""
You are a web agent planner.

Your job is to analyze the current web environment and the agent's progress so far, and then determine the most appropriate next sub-task the agent should perform.

You are given:

1. **Task Goal**: The ultimate objective the user wants to achieve.
2. **Current Page (AXTree)**: A structured textual representation of the current page's accessible elements.
3. **Action History**: The list of actions the agent has performed so far in order.

---

## Important Decision Process

Before proposing the next sub-task, you must first determine whether the task can already be completed.

### 1. Completion Check

Before proposing the next sub-task, determine the user's requested final condition.

A task can be marked as complete only when the current page state and action history provide sufficient evidence that the requested final condition is already satisfied.

Do not mark a task as complete merely because the current page or action history shows an intermediate or prepared state. A locally edited field, selected option, filled form, computed value, identified target, or visible control is not sufficient unless it also provides evidence that the requested final condition has become effective.

If the current interaction creates a pending state that normally requires a further commit, apply, submit, confirm, or save step, the task is not complete until the page state or action history shows that this pending state has been committed or confirmed by the website.

If the requested final condition is to provide an answer, then the task is complete when the answer is explicitly available or can be directly inferred from the current page and action history.

Then:

- If the requested final condition is already satisfied, generate a sub-task that submits the final answer or completion confirmation to the user.
- Otherwise, propose the next atomic step needed to move toward the requested final condition.

### 2. Continue

If the requested final condition is not yet fully satisfied, but further interaction can move the page toward that condition, propose the next atomic step.

---

## Critical Rules

- Avoid unnecessary or repetitive actions.
- Ground your decision strictly in the current AXTree and history.
- Each sub-task must be atomic and contain only one step.

## Tips:
- Specifically, for search task, after filling a search box, if a Search button becomes enabled, the next action should normally be to click the Search button or press Enter. Do not treat filling the box as completing the search unless the page visibly updates with search results.

---

## Output Format

You must output only a valid JSON object. Do not output markdown, code fences, or explanations.

The JSON format must be:

{{
  "decision": "complete" | "continue",
  "sub_task": "One concise instruction describing either a submission or the next action",
  "reason": "Briefly explain why this decision is appropriate"
}}

Requirements:
- If "decision" is "complete", "sub_task" must start with "Submit".
- If "decision" is "continue", "sub_task" must describe the next atomic exploration step.
- The "reason" field should be concise and grounded in the current page and action history.

---

# Current task
Task Goal: {task}
Current Page (AXTree):
{axtree}
Action History:
{self.history.get_history()}
"""

    @staticmethod
    def _build_env_prompt(subtask: str, axtree: str) -> str:
        return f"""
You are an intelligent web agent. Your task is to extract the task-relevant sub-environment from a full Accessibility Tree (AXTree) of a webpage.

Instructions:
- Select all AXTree fragments that are directly or indirectly relevant to the task or sub-task.
- Err on the side of inclusion if uncertain.
- Copy the selected fragments exactly as-is from the original AXTree.
- Do not modify AXNode IDs, node types, labels, values, or attributes.
- The extracted sub-environment must be formatted inside triple backticks.

Output format:
```
[The extracted sub-environment]
```

# Current task
Task:
{subtask}

# Current Page
Accessibility Tree:
{axtree}
"""

    def _build_action_prompt(self, task: str, subtask: str, action_environment: str, url: str) -> str:
        return f"""
You are a web assistant. You will be given web-based tasks that must be completed through specific browser actions.

Information available:
- Final objective: the user's final goal.
- Sub-task: the current atomic step to perform.
- Current web page accessibility tree: the relevant page environment.
- Current URL.
- Action history and failed attempts.

You can interact with the environment using the following actions:
{self.action_set.describe(with_long_description=False)}

Rules:
1. Only issue an action that is valid for the current observation.
2. Only issue one action at a time.

Output format:
### Manipulate element:
[Identify the specific element to interact with.]
### Action:
[Write the exact action command.]
### Action intent:
[Summarize what the action should accomplish.]

Example:
OBSERVATION:
[1] heading '/f/food'
[10] heading '[I ate] Maple Pecan Croissant'
[17] link '204 comments'
URL: http://test.com
OBJECTIVE: Tell me what the top comment on the croissant post says.
Answer:
### Manipulate element:
Click link element [17] labeled '204 comments', which is associated with the croissant post.

### Action:
click("17")

### Action intent:
Navigate to the comments section of the Maple Pecan Croissant post in order to access the top comment.

# Current Task
OBSERVATION:
{action_environment}
URL: {url}
Final Objective: {task}
History Actions:
{self.history.get_history()}
Failure attempts:
{self.history.get_error_descendants()}
Sub-task: {subtask}
ACTION:
"""

    @staticmethod
    def _build_summary_prompt(
        task: str,
        subtask: str,
        action: str,
        intent: str,
        axtree: str,
        element: str,
    ) -> str:
        return f"""
You are a Summarization Agent in a web automation system.

Produce a concise, accurate, context-aware summary of the current step.
Your summary must reflect the page environment, the manipulated element, and the action intent.

Required Output Format (JSON):
{{
  "environment_description": "a brief description of the functional areas of the page",
  "manipulated_element": "the element manipulated by the action",
  "action_summary": "what the action achieves"
}}

# Current task
Task Goal: {task}
Overall Environment of Current Step: {axtree}
Sub-task: {subtask}
Action: {action}
Manipulated Element: {element}
Action Intent: {intent}
"""

    def _truncate_prompt_if_needed(
        self,
        prompt: str,
        environment: str,
        history: Optional[HistoryTree],
        max_tokens: int = 30_000,
    ) -> str:
        result = truncate_prompt(
            base_prompt=prompt,
            history=history,
            environment=environment,
            max_tokens=max_tokens,
            reserve=30,
            model_name=DEFAULT_TOKENIZER_NAME,
            keep_last_n_history=3,
        )
        if not result.is_truncated:
            return prompt

        truncated_history = self.history.get_history()
        if getattr(result, "truncated_hist", False):
            truncated_history = result.truncated_part.get("history", truncated_history)

        truncated_env = environment
        if getattr(result, "truncated_env", False):
            truncated_env = result.truncated_part.get("env", environment)

        return prompt.replace(environment, truncated_env).replace(self.history.get_history(), truncated_history)

    def _extract_planning_output(self, response: str, prompt: str) -> dict[str, Any]:
        data = self.extract_llm_json(agent_name="PlanningAgent", prompt=prompt, output_text=response)
        required = {"decision", "sub_task", "reason"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"PlanningAgent output is missing keys: {missing}")
        if data["decision"] not in {"complete", "continue"}:
            raise ValueError(f"Invalid planning decision: {data['decision']}")
        if not isinstance(data["sub_task"], str) or not data["sub_task"].strip():
            raise ValueError("PlanningAgent sub_task must be a non-empty string.")
        return data

    @staticmethod
    def _extract_code_block(response: str) -> str:
        match = CODE_BLOCK_PATTERN.search(response)
        return match.group(1).strip() if match else response.strip()

    @staticmethod
    def _parse_action_response(response: str) -> dict[str, str]:
        match = ACTION_RESPONSE_PATTERN.search(response)
        if match is None:
            raise ValueError("ActionAgent output does not match the expected format.")
        return {
            "manipulate_element": match.group(1).strip(),
            "action": match.group(2).strip(),
            "action_intent": match.group(3).strip(),
        }

    def extract_llm_json(self, agent_name: str, prompt: str, output_text: str) -> dict[str, Any] | bool:
        match = JSON_OBJECT_PATTERN.search(output_text.strip())
        if match is None:
            self.save_with_timestamp(
                content=f"No JSON object found.\nAgent name: {agent_name}\n\nInput:\n{prompt}\n\nOutput:\n{output_text}",
                directory=self.error_log_dir,
            )
            raise ValueError("No JSON object found in LLM output.")

        json_str = match.group(0).replace("True", "true").replace("False", "false")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error JSON: {json_str}")
            return False

    def _save_record(
        self,
        agent_name: str,
        task: str,
        subtask: Optional[str],
        axtree: str,
        input_text: str,
        output_text: str,
        extracted_part: Any,
        reasoning_content: Optional[str],
        step: int,
        time_cost: Optional[float] = None,
        try_num: Optional[int] = None,
    ) -> dict[str, Any]:
        kwargs = dict(
            step=step,
            agent_name=agent_name,
            task=task,
            subtask=subtask,
            axtree=axtree,
            input=input_text,
            output=output_text,
            extracted_part=extracted_part,
            reasoning_content=reasoning_content,
            timestamp=time.time(),
        )
        if time_cost is not None:
            kwargs["time_cost"] = time_cost
        if try_num is not None:
            kwargs["try_num"] = try_num

        record_kwargs = dict(kwargs)
        try:
            record = AgentIO(**record_kwargs)
        except TypeError:
            record_kwargs.pop("time_cost", None)
            record_kwargs.pop("try_num", None)
            record = AgentIO(**record_kwargs)

        if self.logger is not None:
            self.logger.log(record)

        record_dict = asdict(record)
        if time_cost is not None:
            record_dict["time_cost"] = time_cost
        if try_num is not None:
            record_dict["try_num"] = try_num
        return record_dict

    @staticmethod
    def _format_think(records: list[dict[str, Any]]) -> str:
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)

    @staticmethod
    def save_with_timestamp(
        content: str,
        directory: str,
        prefix: str = "log",
        time_format: str = "%Y%m%d_%H%M%S",
    ) -> str:
        timestamp = datetime.now().strftime(time_format)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{prefix}_{timestamp}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path


@dataclass
class CustomAgentArgs(AgentArgs):
    """AgentLab argument wrapper for constructing ``CustomAgent`` instances."""

    agent_name: str = "BasicAgent"
    temperature: float = 0
    use_chain_of_thought: bool = False
    chat_model_args: "BaseModelArgs" = None
    exp_dir: Optional[str] = None
    logger: Optional[InteractionLogger] = None
    llm_api_base: str = DEFAULT_LLM_API_BASE
    llm_api_key: str = DEFAULT_LLM_API_KEY
    error_log_dir: str = DEFAULT_ERROR_LOG_DIR

    def make_agent(self) -> bgym.Agent:
        return CustomAgent(
            temperature=self.temperature,
            logger=self.logger,
            llm_api_base=self.llm_api_base,
            llm_api_key=self.llm_api_key,
            error_log_dir=self.error_log_dir,
        )
