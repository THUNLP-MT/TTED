from __future__ import annotations

import json
import os
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


DEFAULT_LLM_API_KEY = os.getenv("EVAL_OPENAI_API_KEY", "EMPTY")
DEFAULT_LLM_API_BASE = os.getenv("EVAL_OPENAI_API_BASE", "EMPTY")
DEFAULT_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5-nano")
DEFAULT_ERROR_LOG_DIR = os.getenv("AGENT_ERROR_LOG_DIR", "./logs/agent_errors")
DEFAULT_TOKENIZER_NAME = os.getenv("AGENT_TOKENIZER_NAME", "utils/tokenizer")
DEFAULT_MAX_TOKENS = int(os.getenv("AGENT_MAX_PROMPT_TOKENS", "30000"))
DEFAULT_ACTION_RETRY = int(os.getenv("AGENT_ACTION_RETRY", "5"))
DEFAULT_JUDGE_RETRY = int(os.getenv("AGENT_JUDGE_RETRY", "5"))
DEFAULT_SUMMARY_RETRY = int(os.getenv("AGENT_SUMMARY_RETRY", "5"))

NO_HISTORY_MESSAGE = "No history, this is your first step."
NO_PARENT_HISTORY_MESSAGE = "No history, this is the first step."
NO_FAILURE_MESSAGE = "This is your first attempt."
NO_OP_ACTION = "None"

ACTION_RESPONSE_PATTERN = re.compile(
    r"### Manipulate element:\s*(.*?)\s*### Action:\s*(.*?)\s*### Action intent:\s*(.*)",
    flags=re.DOTALL,
)
CODE_BLOCK_PATTERN = re.compile(r"```(.*?)```", flags=re.DOTALL)
JSON_PATTERN = re.compile(r"\{[\s\S]*?\}")


@dataclass
class HistoryNode:
    """A node in the agent history tree."""

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
    """Maintain the accepted trajectory and failed alternative attempts."""

    def __init__(self) -> None:
        self.root = HistoryNode()
        self.current = self.root
        self.nodes: dict[str, HistoryNode] = {self.root.id: self.root}

    def add_empty_node(self) -> HistoryNode:
        """Create an empty child node and move the current pointer to it."""
        new_node = HistoryNode(parent=self.current)
        self.current.add_child(new_node)
        self.current = new_node
        self.nodes[new_node.id] = new_node
        return new_node

    def backtrack(self) -> None:
        """Move the current pointer back to the parent node."""
        if self.current.parent is None:
            raise RuntimeError("Cannot backtrack because the current node is already the root.")
        self.current = self.current.parent

    def get_current_node(self) -> HistoryNode:
        return self.current

    def get_history(self) -> str:
        return self._format_history_until(self.current, empty_message=NO_HISTORY_MESSAGE)

    def parent_history(self) -> str:
        return self._format_history_until(self.current.parent, empty_message=NO_PARENT_HISTORY_MESSAGE)

    def get_error_descendants(self) -> str:
        errors = [child.error_info for child in self.current.children if child.is_error and child.error_info]
        return "\n".join(errors) if errors else NO_FAILURE_MESSAGE

    def _format_history_until(self, node: Optional[HistoryNode], empty_message: str) -> str:
        if node is None or node.action is None:
            return empty_message

        rows: list[str] = []
        while node is not None and node.action is not None:
            rows.append(
                "url: {url}, action: {action}, action summary: {summary}".format(
                    url=node.url,
                    action=node.action,
                    summary=node.intent,
                )
            )
            node = node.parent

        rows.reverse()
        return "\n".join(f"Step {idx}. {row}" for idx, row in enumerate(rows, start=1))


class CustomAgent(Agent):
    """An action+summary agent with an in-step action judge.

    The flow follows the provided action+summary+judge version:
    1. Generate one action from the current page.
    2. Immediately judge whether the action is useful based on the current page.
    3. If accepted, summarize and return the action for execution.
    4. If rejected, record the failed attempt, backtrack, and return ``None``.
    """

    def __init__(
        self,
        temperature: float,
        logger: Optional[InteractionLogger],
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
        self.last_false_page: Optional[str] = None

        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.error_log_dir = error_log_dir
        self._client: Optional[OpenAI] = None

        self.action_retry_num = DEFAULT_ACTION_RETRY
        self.judge_retry_num = DEFAULT_JUDGE_RETRY
        self.summary_retry_num = DEFAULT_SUMMARY_RETRY

    def get_action(self, obs: Any) -> tuple[str, bgym.AgentInfo]:
        """Generate an action, judge it in the current state, then return it only if accepted."""
        if "RootWebArea ''" in obs.get("axtree_txt", ""):
            print("blank page")
            return "go_forward()", bgym.AgentInfo()

        think: list[str] = []
        self._initialize_root_if_needed(obs)
        self.counter += 1

        action_prompt = self._build_action_prompt(obs)
        action_prompt = self._truncate_action_prompt_if_needed(action_prompt, obs)
        reasoning_content, action_response, action_info = self._query_action_with_retries(action_prompt)
        action = action_info["action"]

        action_record = self._save(
            agent_name="ActionAgent",
            task=obs["goal"],
            subtask=None,
            axtree=obs["axtree_txt"],
            input_text=action_prompt,
            output_text=action_response,
            extracted_part=action_info,
            reasoning_content=reasoning_content,
            step=self.counter,
        )
        think.append(action_record)

        node = self._append_action_to_history(obs, action_info)

        judge_result, judge_cot, judge_record = self._action_judge(
            action_intent=action_info["action_intent"],
            history=self.history.parent_history(),
            action=action,
            element=action_info["manipulate_element"],
            task_goal=obs["goal"],
            env=obs["axtree_txt"],
        )
        think.append(judge_record)

        if judge_result:
            self.last_false_page = None
            summary, summary_record = self.summary_agent(
                task=obs["goal"],
                action=action,
                intent=action_info["action_intent"],
                judge=judge_cot,
                axtree=obs["axtree_txt"],
                element=action_info["manipulate_element"],
            )
            summary_text = self._format_summary_for_history(summary)
            node.intent = summary_text
            think.append(summary_record)
            print(f"{self.counter}: {action}, success")
        else:
            false_message, false_message_record = self.false_message_agent(
                task=obs["goal"],
                action=action,
                action_intent=action_info["action_intent"],
                judge_cot=judge_cot,
                axtree=obs["axtree_txt"],
            )
            think.append(false_message_record)
            node.mark_as_error(false_message)
            self.history.backtrack()
            print(f"{self.counter}: {action}, fail")
            action = NO_OP_ACTION

        return action, bgym.AgentInfo(think="\n".join(think))

    def _initialize_root_if_needed(self, obs: Any) -> None:
        if self.counter != 0:
            return
        self.history.root.axtree = obs["axtree_txt"]
        self.history.root.url = obs["url"]

    def _append_action_to_history(self, obs: Any, action_info: dict[str, str]) -> HistoryNode:
        self.history.add_empty_node()
        node = self.history.get_current_node()
        node.action = action_info["action"]
        node.element = action_info["manipulate_element"]
        node.intent = action_info["action_intent"]
        node.axtree = obs["axtree_txt"]
        node.url = obs["url"]
        return node

    def _query_action_with_retries(self, prompt: str) -> tuple[Optional[str], str, dict[str, str]]:
        last_reasoning_content: Optional[str] = None
        last_response = ""

        for _ in range(self.action_retry_num):
            try:
                last_reasoning_content, last_response = self.llm(prompt)
                action_info = self._parse_action_response(last_response)
                return last_reasoning_content, last_response, action_info
            except Exception:
                continue

        self.save_with_timestamp(
            "(ActionAgent) The response does not match the expected format.\n"
            f"Input:\n{prompt}\n\nOutput:\n{last_response}",
            self.error_log_dir,
        )
        raise ValueError("ActionAgent output does not match the expected format.")

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

    def llm(self, prompt: str) -> tuple[Optional[str], str]:
        """Call an OpenAI-compatible chat completion endpoint."""
        if self._client is None:
            self._client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_api_base)

        model = self.llm_model_name
        if not model:
            model = self._client.models.list().data[0].id
            self.llm_model_name = model

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=5000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        content = message.content or ""

        if response.choices[0].finish_reason == "length":
            truncated_reasoning = (
                content + "\n</think>" if reasoning_content is None else f"<think>\n{reasoning_content}\n</think>"
            )
            second_response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt + "\n" + truncated_reasoning}],
                temperature=self.temperature,
                max_tokens=5000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return truncated_reasoning, second_response.choices[0].message.content or ""

        return reasoning_content, content

    def _build_action_prompt(self, obs: Any) -> str:
        return f"""
You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's task: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The history actions' summary: These are the history actions you have done, including actions and its intent.

You can interact with the environment using the following actions:
{self.action_set.describe(with_long_description=False)}

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.

Your output should include:
- Clearly explain what the action is trying to achieve.
- Clearly state which element is being operated on, including element ID and label/type.
- Output the final action command that should be executed.

Output format:
### Manipulate element:
[Identify the specific element that the agent will interact with.]
### Action:
[Write the exact action command that should be executed on the page.]
### Action intent:
[Summarize what the action is trying to accomplish in the context of the task.]

Here is an example of how to use the bid action:

click('314')

Example:

OBSERVATION:
[1] heading '/f/food'
[10] heading '[I ate] Maple Pecan Croissant'
    [11] link '[I ate] Maple Pecan Croissant'
[17] link '204 comments'
URL: http://test.com
OBJECTIVE: Tell me what the top comment on the croissant post says.
ACTION:
### Manipulate element:
Click on link element [17] labeled '204 comments', which is associated with the croissant post.

### Action:
click("17")

### Action intent:
Navigate to the comments section of the Maple Pecan Croissant post in order to access the top comment.

# Current Task
OBSERVATION:
{obs["axtree_txt"]}
URL: {obs["url"]}
TASK: {obs["goal"]}
History Actions' Summary:
{self.history.get_history()}
Failure attempts:
{self.history.get_error_descendants()}
ACTION:
"""

    def _truncate_action_prompt_if_needed(self, prompt: str, obs: Any) -> str:
        result = truncate_prompt(
            base_prompt=prompt,
            history=self.history,
            environment=obs["axtree_txt"],
            max_tokens=DEFAULT_MAX_TOKENS,
            reserve=30,
            model_name=DEFAULT_TOKENIZER_NAME,
            keep_last_n_history=3,
        )
        if not result.is_truncated:
            return prompt

        truncated_history = result.truncated_part["history"] if result.truncated_hist else self.history.get_history()
        truncated_env = result.truncated_part["env"] if result.truncated_env else obs["axtree_txt"]

        return f"""
You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's task: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The history actions' summary: These are the history actions you have done, including actions and its intent.

You can interact with the environment using the following actions:
{self.action_set.describe(with_long_description=False)}

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.

Your output should include:
- Clearly explain what the action is trying to achieve.
- Clearly state which element is being operated on, including element ID and label/type.
- Output the final action command that should be executed.

Output format:
### Manipulate element:
[Identify the specific element that the agent will interact with.]
### Action:
[Write the exact action command that should be executed on the page.]
### Action intent:
[Summarize what the action is trying to accomplish in the context of the task.]

Here is an example of how to use the bid action:

click('314')

# Current Task
OBSERVATION:
{truncated_env}
URL: {obs["url"]}
TASK: {obs["goal"]}
History Actions' Summary:
{truncated_history}
Failure attempts:
{self.history.get_error_descendants()}
ACTION:
"""

    def _action_judge(
        self,
        action_intent: str,
        history: str,
        action: str,
        element: str,
        task_goal: str,
        env: str,
    ) -> tuple[bool, str, str]:
        prompt = f"""
You are an intelligent evaluation module for a web automation agent.
Your task is to determine whether a specific action taken by the agent has successfully fulfilled its intended purpose.
You will be provided with:
1. Overall Task Goal — the agent's high-level objective.
2. Overall Environment of Current Step — the full environment.
3. Action History — a list of previously executed steps.
4. Executed Action — the specific action the agent just performed.
5. Detailed Element — the DOM element involved in the action.
6. Action Intent — the natural-language description of what the agent wanted to achieve with the action.

You must determine whether the executed action successfully fulfilled its intent.
- Based on the action and environment, determine whether the action can contribute positively toward achieving the overall task goal.
- If the action contributes positively toward achieving the overall task goal, count it as successful.
- Provide a brief explanation of why the action succeeded or failed.

Required Output Format (JSON)
{{
  "action_success": true or false,
  "action_reasoning": "why the action succeeded or failed"
}}

Current Evaluate
Overall Task Goal:
{task_goal}

Overall Environment of Current Step:
{env}

Previous Steps:
{history}

Executed Action:
{action}

Detailed Element:
{element}

Action Intent:
{action_intent}
"""
        prompt = self._truncate_generic_prompt_if_needed(prompt, env, self.history.parent_history())

        result: Any = False
        last_reasoning_content: Optional[str] = None
        last_response = ""
        for _ in range(self.judge_retry_num):
            last_reasoning_content, last_response = self.llm(prompt)
            result = self.extract_llm_json(last_response.strip(), prompt, agent_name="ActionJudgeAgent")
            if result:
                break

        if not result:
            raise RuntimeError("Failed to extract JSON from ActionJudgeAgent output.")

        record = self._save(
            agent_name="ActionJudgeAgent",
            task=task_goal,
            subtask=None,
            axtree=env,
            input_text=prompt,
            output_text=last_response,
            extracted_part=result,
            reasoning_content=last_reasoning_content,
            step=self.counter,
        )
        return bool(result.get("action_success", False)), last_response.strip(), record

    def summary_agent(
        self,
        task: str,
        action: str,
        intent: str,
        judge: str,
        axtree: str,
        element: str,
    ) -> tuple[dict[str, Any], str]:
        prompt = f"""
You are a Summarization Agent in a web automation system.

Your role is to produce a concise, accurate, and context-aware summary of the current step.
Your summary must reflect (1) the current page environment, (2) what element was manipulated, and (3) the action's intent.

You are given:
1. Task Goal: The overall objective of the task.
2. Overall Environment of Current Step: The full environment of current step.
3. Action: The action will be executed.
4. Manipulated Element: The element that is manipulated by the action.
5. Action Intent: A description of what the action is trying to achieve.
6. Judge Result: The action has been accepted by the judge.

Required Output Format (JSON)
{{
  "environment_description": "a brief description of the functional areas of the page",
  "manipulated_element": "the element that is manipulated by the action",
  "action_summary": "a description of what the action achieves"
}}

# Current task
Task Goal: {task}
Overall Environment of Current Step: {axtree}
Action: {action}
Manipulated Element: {element}
Action Intent: {intent}
Judge Result: {judge}
"""
        prompt = self._truncate_generic_prompt_if_needed(prompt, axtree, None)

        result: Any = False
        last_reasoning_content: Optional[str] = None
        last_response = ""
        for _ in range(self.summary_retry_num):
            last_reasoning_content, last_response = self.llm(prompt)
            result = self.extract_llm_json(last_response.strip(), prompt, agent_name="SummaryAgent")
            if result:
                break

        if not result:
            raise RuntimeError("Failed to extract JSON from SummaryAgent output.")

        record = self._save(
            agent_name="SummaryAgent",
            task=task,
            subtask=None,
            axtree=axtree,
            input_text=prompt,
            output_text=last_response,
            extracted_part=result,
            reasoning_content=last_reasoning_content,
            step=self.counter,
        )
        return result, record

    def false_message_agent(
        self,
        task: str,
        action: str,
        action_intent: str,
        judge_cot: str,
        axtree: str,
    ) -> tuple[str, str]:
        prompt = f"""
You are a failure explanation agent for a web automation system.

Your job is to analyze why a given action failed to achieve its intended effect, and to summarize the issue clearly along with suggestions to avoid or fix it in the future.

You are given:
1. Task Goal: The overall goal the agent is trying to accomplish.
2. Action: The action that was executed.
3. Action Intent: What the agent was trying to achieve with that action.
4. Failure Analysis: A reasoning description of why the action failed.

Output format:
Failure Summary:
[Describe what went wrong.]

Cause Analysis:
[Explain the likely root cause.]

# Current task
Task Goal: {task}
Action: {action}
Action Intent: {action_intent}
Failure Analysis: {judge_cot}
"""
        reasoning_content, response = self.llm(prompt)
        false_message = (
            f"Action: {action}\n"
            f"Action intent:\n{action_intent}\n"
            f"Analysis:\n{response}\n"
            "Conclusion: bad attempt."
        )
        record = self._save(
            agent_name="FalseMessageAgent",
            task=task,
            subtask=None,
            axtree=axtree,
            input_text=prompt,
            output_text=response,
            extracted_part=false_message,
            reasoning_content=reasoning_content,
            step=self.counter,
        )
        return false_message, record

    def _truncate_generic_prompt_if_needed(
        self,
        prompt: str,
        environment: str,
        history_text: Optional[str],
    ) -> str:
        result = truncate_prompt(
            base_prompt=prompt,
            history=self.history if history_text is not None else None,
            environment=environment,
            max_tokens=DEFAULT_MAX_TOKENS,
            reserve=30,
            model_name=DEFAULT_TOKENIZER_NAME,
            keep_last_n_history=3,
        )
        if not result.is_truncated:
            return prompt

        truncated_env = result.truncated_part["env"] if result.truncated_env else environment
        truncated_history = result.truncated_part["history"] if result.truncated_hist else history_text
        prompt = prompt.replace(environment, truncated_env)
        if history_text is not None and truncated_history is not None:
            prompt = prompt.replace(history_text, truncated_history)
        return prompt

    @staticmethod
    def _format_summary_for_history(summary: dict[str, Any]) -> str:
        return (
            f"environment_description: {summary.get('environment_description', '')}, "
            f"manipulated_element: {summary.get('manipulated_element', '')}, "
            f"action_summary: {summary.get('action_summary', '')}"
        )

    def extract_llm_json(self, output_text: str, prompt: str, agent_name: str) -> Any:
        match = JSON_PATTERN.search(output_text)
        if not match:
            self.save_with_timestamp(
                f"No JSON object was found.\nAgent name: {agent_name}\n\nInput:\n{prompt}\n\nOutput:\n{output_text}",
                self.error_log_dir,
            )
            raise ValueError("No JSON object was found in the LLM output.")

        json_str = match.group(0).replace("True", "true").replace("False", "false")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error JSON: {json_str}")
            return False

    def _save(
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
    ) -> str:
        record = AgentIO(
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
        if self.logger is not None:
            self.logger.log(record)
        return json.dumps(asdict(record), ensure_ascii=False)

    def save_with_timestamp(
        self,
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
