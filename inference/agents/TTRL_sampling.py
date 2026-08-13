from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, List, Optional

import bgym
from agentlab.agents.agent_args import AgentArgs
from bgym import Agent
from openai import OpenAI

from utils.io_record import AgentIO, InteractionLogger
from utils.prompt_packer import truncate_prompt
from utils.subenv_expand import expand_subenv_once_to_text_wide_with_metrics


DEFAULT_LLM_API_KEY = os.getenv("EVAL_OPENAI_API_KEY", "EMPTY")
DEFAULT_LLM_API_BASE = os.getenv("EVAL_OPENAI_API_BASE", "EMPTY")
DEFAULT_ERROR_LOG_DIR = os.getenv("AGENT_ERROR_LOG_DIR", "./logs/agent_errors")
DEFAULT_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "gpt-5-nano")


class HistoryNode:
    def __init__(self, parent: Optional['HistoryNode'] = None):
        self.id = str(uuid.uuid4())
        self.action: Optional[str] = None
        self.element: Optional[str] = None
        self.intent: Optional[str] = None
        self.url: Optional[str] = None
        self.axtree: Optional[str] = None
        self.subtask: Optional[str] = None
        self.parent = parent
        self.children: List['HistoryNode'] = []
        self.is_error = False
        self.error_info = None
        self.subenv: Optional[str] = None

    def add_child(self, child: 'HistoryNode'):
        self.children.append(child)

    def __repr__(self):
        return f"HistoryNode(id={self.id[:6]}, action={self.action}, url={self.url})"

    def get_action(self):
        return self.action

    def get_element(self):
        return self.element

    def get_url(self):
        return self.url

    def get_intent(self):
        return self.intent

    def get_axtree(self):
        return self.axtree

    def get_subtask(self):
        return self.subtask

    def get_subenv(self):
        return self.subenv

    def get_error_info(self):
        return self.error_info

    def get_error_state(self):
        return self.is_error

    def set_action(self, action):
        self.action = action

    def set_element(self, element):
        self.element = element

    def set_intent(self, intent):
        self.intent = intent

    def set_url(self, url):
        self.url = url

    def set_axtree(self, axtree):
        self.axtree = axtree

    def set_subtask(self, subtask):
        self.subtask = subtask

    def set_subenv(self, subenv):
        self.subenv = subenv

    def set_error_info(self, error_info):
        self.error_info = error_info

    def set_as_error(self):
        self.is_error = True

class HistoryTree:
    def __init__(self):
        self.root = HistoryNode()
        self.current = self.root
        self.nodes = {self.root.id: self.root}

    def add_empty_node(self) -> HistoryNode:
        """Create an empty child node and move the current pointer to it."""
        new_node = HistoryNode(parent=self.current)
        self.current.add_child(new_node)
        self.current = new_node
        self.nodes[new_node.id] = new_node
        return new_node

    def backtrack(self):
        if self.current.parent:
            self.current = self.current.parent
        else:
            raise RuntimeError("Cannot backtrack further because the current node is already the root.")

    def get_current_path(self) -> List[HistoryNode]:
        path = []
        node = self.current
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_current_node(self):
        return self.current

    def get_history(self) -> str:
        history = []
        node = self.current
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"url: {node.get_url()}, action: {node.get_action()}, action summary: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        history_t = []
        idx = 1
        for h in history:
            h = f"Step {idx}. {h}"
            history_t.append(h)
            idx += 1
        history = "\n".join(history_t)
        return history

    def get_n_history(self, n) -> str:
        history = []
        node = self.current
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"url: {node.get_url()}, action: {node.get_action()}, action summary: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        history_t = []
        idx = 1
        for h in history:
            h = f"Step {idx}. {h}"
            history_t.append(h)
            idx += 1
        history = history_t[-n:]
        history = "\n".join(history)
        return history
    def get_any_node_history(self, node) -> str:
        history = []
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"url: {node.get_url()}, action: {node.get_action()}, action summary: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        history_t = []
        idx = 1
        for h in history:
            h = f"Step {idx}. {h}"
            history_t.append(h)
            idx += 1
        history = "\n".join(history_t)
        return history

    def parent_history(self) -> str:
        parent_node = self.current.parent
        if parent_node.get_action() == None:
            return "No history, this is the first step."
        else:
            return self.get_any_node_history(parent_node)

    def get_error_descendants(self):
        # Collect failed attempts among the current node's children.
        errors = []
        for child in self.current.children:
            if child.is_error:
                errors.append(child.error_info)
        if len(errors) == 0:
            return "This is your first attempt."
        else:
            errors = "\n".join(errors)
        return errors


class CustomAgent(Agent):
    def __init__(
        self,
        temperature: float,
        logger: Optional["InteractionLogger"],
        llm_api_base: str = DEFAULT_LLM_API_BASE,
        llm_api_key: str = DEFAULT_LLM_API_KEY,
        llm_model_name: str = DEFAULT_MODEL_NAME,
        error_log_dir: str = DEFAULT_ERROR_LOG_DIR,
    ):
        """Initialize the web agent, action space, history tree, and runtime paths."""
        self.temperature = temperature
        self.action_set = bgym.HighLevelActionSet(["webarena"], multiaction=False)
        self.counter = 0
        self.judge_stage = False
        self.history = HistoryTree()
        self.logger = logger
        self.last_false_page = None
        self.error_log_path = error_log_dir
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.sampling_num = int(os.getenv("AGENT_SAMPLING_NUM", "4"))
        self.action_retry_num = int(os.getenv("AGENT_ACTION_RETRY", "5"))

    def get_action(self, obs: Any) -> tuple[str, dict]:
        """Generate four sampled actions, select one, and then run the normal summary step."""
        if "RootWebArea ''" in obs["axtree_txt"]:
            print("blank page")
            return "go_forward()", bgym.AgentInfo()

        think: list[str] = []
        if self.counter == 0:
            root_node = self.history.root
            root_node.set_axtree(obs["axtree_txt"])
            root_node.set_url(obs["url"])
        self.counter += 1

        prompt = self._build_action_prompt(obs)
        prompt = self._truncate_action_prompt_if_needed(prompt, obs)

        candidates: list[dict[str, str]] = []
        sampling_records: list[str] = []

        for sample_idx in range(self.sampling_num):
            reasoning_content, response, ans_dict = self._query_action_with_retries(prompt)
            candidates.append(ans_dict)

            sampling_record = self._save(
                agent_name="Sampling",
                task=obs["goal"],
                subtask=None,
                axtree=obs["axtree_txt"],
                input=prompt,
                output=response,
                extracted_part={"try_num": sample_idx, **ans_dict},
                reasoning_content=reasoning_content,
                step=self.counter,
            )
            sampling_records.append(sampling_record)
            think.append(sampling_record)

        sampling_idx = random.randint(0, len(candidates) - 1)
        ans_dict = candidates[sampling_idx]
        action = ans_dict["action"]
        print(action)

        action_record_dict = json.loads(sampling_records[sampling_idx])
        action_record_dict["agent_name"] = "ActionAgent"
        action_record_dict["extracted_part"] = ans_dict
        think.append(json.dumps(action_record_dict, ensure_ascii=False))

        self.history.add_empty_node()
        current_node = self.history.get_current_node()
        current_node.set_action(action)
        current_node.set_element(ans_dict["manipulate_element"])
        current_node.set_intent(ans_dict["action_intent"])
        current_node.set_axtree(obs["axtree_txt"])
        current_node.set_url(obs["url"])

        summary, summary_record = self.summary_agent(
            obs["goal"],
            action,
            ans_dict["action_intent"],
            obs["axtree_txt"],
            ans_dict["manipulate_element"],
        )
        current_node.set_intent(summary)
        think.append(summary_record)

        return action, bgym.AgentInfo(think="\n".join(think))

    def _build_action_prompt(self, obs: Any) -> str:
        """Build the standard action-generation prompt used by the action+summary agent."""
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
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.

Your output should be included:
- Clearly explain what the action is trying to achieve (intent)
- Clearly state which element is being operated on (element ID + label/type)
- Output the final action command that should be executed

Output format:
### Manipulate element:
[Identify the specific element that the agent will interact with.]
### Action:
[Write the exact action command that should be executed on the page.]
### Action intent:
[Summarize what the action is trying to accomplish in the context of the task.]

Here is an example of how to use the bid action:

click('314')

Here are some examples:
# Example
Example 1/3:

OBSERVATION:
[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
[1749] StaticText '$279.49'
[1757] button 'Add to Cart'
[1760] button 'Add to Wish List'
[1761] button 'Add to Compare'
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine?
ACTION:
### Manipulate element:
Read the static text element [1749] which displays the price as '$279.49', and then send it to user.

### Action:
send_msg_to_user("$279.49")

### Action intent:
Extract the price of the HP Inkjet Fax Machine as requested in the task objective.

Example 2/3:

OBSERVATION:
[204] heading '/f/food'
[593] heading '[homemade] Obligatory Halloween Pumpkin Loaf!'
    [942] link '[homemade] Obligatory Halloween Pumpkin Loaf!'
[945] StaticText 'Submitted by '
[30] link 'kneechalice' expanded: False
[1484] StaticText 't3_yid9lu'
[949] time 'October 31, 2022 at 10:10:03 AM EDT'
    [1488] StaticText '1 year ago'
[1489] link '45 comments'
[605] heading '[I ate] Maple Pecan Croissant'
    [963] link '[I ate] Maple Pecan Croissant'
[966] StaticText 'Submitted by '
[37] link 'AccordingtoJP' expanded: False
[1494] StaticText 't3_y3hrpn'
[970] time 'October 13, 2022 at 10:41:09 PM EDT'
    [1498] StaticText '1 year ago'
[1499] link '204 comments'
URL: http://reddit.com
OBJECTIVE: Tell me what the top comment on the croissant post says.
Answer:
### Manipulate element:
Click on link element [1499] labeled '204 comments', which is associated with the croissant post.

### Action:
click("1499")

### Action intent:
Navigate to the comments section of the Maple Pecan Croissant post in order to access the top comment.

Example 3/3:

OBSERVATION:
[42] link 'My account'
[43] link 'Logout'
[44] link 'Publish Ad'
[25] heading 'What are you looking for today?'
[143] StaticText 'Keyword'
[81] textbox 'e.g., a blue used car' required: False
[146] StaticText 'Category'
[28] heading 'Latest Listings'
[86] link 'Atlas Powered Audio System w/ Tripod'
    [176] img 'Atlas Powered Audio System w/ Tripod'
[511] StaticText '150.00 $'
[88] link 'Neptune Gaming Console'
    [178] img 'Neptune Gaming Console'
[515] StaticText '350.00 $'
URL: http://classifieds.com
OBJECTIVE: Help me find the cheapest dark colored guitar.
Answer:
### Manipulate element:
Fill the search textbox element [81] labeled 'e.g., a blue used car' with the keyword "guitar".

### Action:
fill("81", "guitar")

### Action intent:
Search for guitars on the website to help find the cheapest dark-colored guitar as described in the task.

# Current Task
OBSERVATION:
{obs['axtree_txt']}
TASK: {obs["goal"]}
History Actions' Summary:
{self.history.get_history()}
ACTION:
"""

    def _truncate_action_prompt_if_needed(self, prompt: str, obs: Any) -> str:
        """Apply the same prompt truncation policy used by the original action+summary agent."""
        res = truncate_prompt(
            base_prompt=prompt,
            history=self.history,
            environment=obs["axtree_txt"],
            max_tokens=30000,
            reserve=30,
            model_name="utils/tokenizer",
            keep_last_n_history=3,
        )
        if not res.is_truncated:
            return prompt

        truncated_history = res.truncated_part["history"] if res.truncated_hist else self.history.get_history()
        truncated_env = res.truncated_part["env"] if res.truncated_env else obs["axtree_txt"]

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
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.

Your output should be included:
- Clearly explain what the action is trying to achieve (intent)
- Clearly state which element is being operated on (element ID + label/type)
- Output the final action command that should be executed

Output format:
### Manipulate element:
[Identify the specific element that the agent will interact with.]
### Action:
[Write the exact action command that should be executed on the page.]
### Action intent:
[Summarize what the action is trying to accomplish in the context of the task.]

Here is an example of how to use the bid action:

click('314')

Here are some examples:
# Example
Example 1/3:

OBSERVATION:
[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
[1749] StaticText '$279.49'
[1757] button 'Add to Cart'
[1760] button 'Add to Wish List'
[1761] button 'Add to Compare'
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine?
ACTION:
### Manipulate element:
Read the static text element [1749] which displays the price as '$279.49', and then send it to user.

### Action:
send_msg_to_user("$279.49")

### Action intent:
Extract the price of the HP Inkjet Fax Machine as requested in the task objective.

Example 2/3:

OBSERVATION:
[204] heading '/f/food'
[593] heading '[homemade] Obligatory Halloween Pumpkin Loaf!'
    [942] link '[homemade] Obligatory Halloween Pumpkin Loaf!'
[945] StaticText 'Submitted by '
[30] link 'kneechalice' expanded: False
[1484] StaticText 't3_yid9lu'
[949] time 'October 31, 2022 at 10:10:03 AM EDT'
    [1488] StaticText '1 year ago'
[1489] link '45 comments'
[605] heading '[I ate] Maple Pecan Croissant'
    [963] link '[I ate] Maple Pecan Croissant'
[966] StaticText 'Submitted by '
[37] link 'AccordingtoJP' expanded: False
[1494] StaticText 't3_y3hrpn'
[970] time 'October 13, 2022 at 10:41:09 PM EDT'
    [1498] StaticText '1 year ago'
[1499] link '204 comments'
URL: http://reddit.com
OBJECTIVE: Tell me what the top comment on the croissant post says.
Answer:
### Manipulate element:
Click on link element [1499] labeled '204 comments', which is associated with the croissant post.

### Action:
click("1499")

### Action intent:
Navigate to the comments section of the Maple Pecan Croissant post in order to access the top comment.

Example 3/3:

OBSERVATION:
[42] link 'My account'
[43] link 'Logout'
[44] link 'Publish Ad'
[25] heading 'What are you looking for today?'
[143] StaticText 'Keyword'
[81] textbox 'e.g., a blue used car' required: False
[146] StaticText 'Category'
[28] heading 'Latest Listings'
[86] link 'Atlas Powered Audio System w/ Tripod'
    [176] img 'Atlas Powered Audio System w/ Tripod'
[511] StaticText '150.00 $'
[88] link 'Neptune Gaming Console'
    [178] img 'Neptune Gaming Console'
[515] StaticText '350.00 $'
URL: http://classifieds.com
OBJECTIVE: Help me find the cheapest dark colored guitar.
Answer:
### Manipulate element:
Fill the search textbox element [81] labeled 'e.g., a blue used car' with the keyword "guitar".

### Action:
fill("81", "guitar")

### Action intent:
Search for guitars on the website to help find the cheapest dark-colored guitar as described in the task.

# Current Task
OBSERVATION:
{truncated_env}
TASK: {obs["goal"]}
History Actions' Summary:
{truncated_history}
ACTION:
"""

    def _query_action_with_retries(self, prompt: str) -> tuple[Any, str, dict[str, str]]:
        """Call the LLM until a valid action-format response is produced."""
        last_response = ""
        last_reasoning_content = None
        for _ in range(self.action_retry_num):
            try:
                last_reasoning_content, last_response = self.llm(prompt)
                ans_dict = self._parse_action_response(last_response)
                return last_reasoning_content, last_response, ans_dict
            except Exception:
                continue

        self.save_with_timestamp(
            f"(Action Agent) The response does not match the expected format.\n"
            f"Agent name: ActionAgent\n\nInput:{prompt}\n\nOutput: {last_response}",
            self.error_log_path,
        )
        raise ValueError("The response does not match the expected format.")

    @staticmethod
    def _parse_action_response(response: str) -> dict[str, str]:
        """Parse the standard action output block."""
        pattern = r"### Manipulate element:\s*(.*?)\s*### Action:\s*(.*?)\s*### Action intent:\s*(.*)"
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            raise ValueError("The response does not match the expected format.")
        return {
            "manipulate_element": match.group(1).strip(),
            "action": match.group(2).strip(),
            "action_intent": match.group(3).strip(),
        }

    def llm(self, prompt: str):
        """Call an OpenAI-compatible chat completion endpoint.

        The API key and base URL are read from the agent configuration. By
        default, they use ``OPENAI_API_KEY`` and ``OPENAI_API_BASE`` environment
        variables, with a local vLLM-compatible endpoint as the fallback.
        """
        client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_api_base)
        model = self.llm_model_name
        # print(self.llm_api_key, self.llm_api_base, self.llm_model_name)
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=5000,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        content = message.content

        if response.choices[0].finish_reason == "length":
            if reasoning_content is None:
                truncated_reasoning = content + "\n</think>"
            else:
                truncated_reasoning = f"<think>\n{reasoning_content}\n</think>"

            new_prompt = prompt + "\n" + truncated_reasoning
            new_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": new_prompt}],
                temperature=self.temperature,
                max_tokens=5000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return truncated_reasoning, new_response.choices[0].message.content

        return reasoning_content, content

    def task_decomp(self, task, html):
        history = self.history.get_history()
        prompt = f"""
You are a web agent planner.

Your job is to analyze the current web environment and the agent's progress so far, and then determine the most appropriate next sub-task the agent should perform.

You are given the following information:

1. **Task Goal**: The ultimate objective the user wants to achieve.
2. **Current Page (AXTree)**: A structured textual representation of the current page's accessible elements.
3. **Action History**: The list of actions the agent has performed so far in order.
4. **Failure Attempts**: A list of previously attempted actions that failed on this page, along with the reasons for their failure.

Your job:
- Reflect on the task goal, the current state of the page, and what has already been tried.
- Identify the most reasonable and necessary next sub-task to move the agent closer to the goal.
- The sub-task should be high-level but specific, and only describe the next immediate step.
    - Do not provide low-level commands such as click('277'). Instead, describe the sub-task using high-level natural language — for example: "Click the link labeled 'The A11Y Project / a11yproject.com' to open the project page."
- The sub-task should be as simple as possible, each containing only one objective and requiring only a single step to complete.
- The sub-task in your answer should be wrapped in ```.

**Output format:**

### Next Sub-task:
```
[One concise, natural language instruction describing the agent's next step.]
```

# Current task
Task Goal: {task}
Current Page (AXTree):
{html}
Action History:
{history}
Failure Attempts:
{self.history.get_error_descendants()}
"""
        res = truncate_prompt(
            base_prompt=prompt,
            history=self.history,
            environment=html,
            max_tokens=30000,
            reserve=30,
            model_name="utils/tokenizer",
            keep_last_n_history=3
        )
        if res.is_truncated:
            if res.truncated_hist:
                truncated_history = res.truncated_part["history"]
            else:
                truncated_history = self.history.get_history()
            if res.truncated_env:
                truncated_env = res.truncated_part["env"]
            else:
                truncated_env = html
            prompt = f"""
You are a web agent planner.

Your job is to analyze the current web environment and the agent's progress so far, and then determine the most appropriate next sub-task the agent should perform.

You are given the following information:

1. **Task Goal**: The ultimate objective the user wants to achieve.
2. **Current Page (AXTree)**: A structured textual representation of the current page's accessible elements.
3. **Action History**: The list of actions the agent has performed so far in order.
4. **Failure Attempts**: A list of previously attempted actions that failed on this page, along with the reasons for their failure.

Your job:
- Reflect on the task goal, the current state of the page, and what has already been tried.
- Identify the most reasonable and necessary next sub-task to move the agent closer to the goal.
- The sub-task should be high-level but specific, and only describe the next immediate step.
    - Do not provide low-level commands such as click('277'). Instead, describe the sub-task using high-level natural language — for example: "Click the link labeled 'The A11Y Project / a11yproject.com' to open the project page."
- The sub-task should be as simple as possible, each containing only one objective and requiring only a single step to complete.
- The sub-task in your answer should be wrapped in ```.

**Output format:**

### Next Sub-task:
```
[One concise, natural language instruction describing the agent's next step.]
```

# Current task
Task Goal: {task}
Current Page (AXTree):
{truncated_env}
Action History:
{truncated_history}
Failure Attempts:
{self.history.get_error_descendants()}
"""

        reasoning_content, response = self.llm(prompt)
        sub_task = self._parser(response).strip()
        record = self._save(
            agent_name="PlanningAgent",
            task=task,
            subtask=sub_task,
            axtree=html,
            input=prompt,
            output=response,
            extracted_part=sub_task,
            reasoning_content=reasoning_content,
            step=self.counter
        )
        return sub_task, response, record

    def env_decomp(self, task, html, goal):
        prompt = f'''
You are an intelligent web agent. Your task is to extract the task-relevant sub-environment from a full **Accessibility Tree (AX Tree)** of a webpage.

Given:
- A user task description.
- The full AX Tree text representation of the current web page.

Instructions:
- Select all AX Tree fragments (i.e., nodes and their subtrees) that are directly or indirectly relevant to the task or sub-task.
- Err on the side of inclusion: **if you are unsure whether an element is needed, include it.**
- You must **copy the selected fragments exactly as-is** from the original AX Tree.
- **Do not modify** any content — including the AXNode ID, node type, labels, values, or attributes.
- AXNode IDs (e.g., `[163]`) must **match the original AX Tree exactly**.
- You may extract multiple non-contiguous branches if needed.
- Capture as much of the environment as possible to avoid omissions or errors.

The extracted sub-environment must be formatted inside triple backticks.

---

### Output format:

```
[The extracted sub-environment]
```


# Current task:
Task:
{task}

# Current Page
Accessibility Tree:
{html}
        '''

        res = truncate_prompt(
            base_prompt=prompt,
            history=None,
            environment=html,
            max_tokens=30000,
            reserve=30,
            model_name="utils/tokenizer",
            keep_last_n_history=3
        )
        if res.is_truncated:
            if res.truncated_hist:
                truncated_history = res.truncated_part["history"]
            else:
                truncated_history = self.history.get_history()
            if res.truncated_env:
                truncated_env = res.truncated_part["env"]
            else:
                truncated_env = html
            prompt = f'''
You are an intelligent web agent. Your task is to extract the task-relevant sub-environment from a full **Accessibility Tree (AX Tree)** of a webpage.

Given:
- A user task description.
- The full AX Tree text representation of the current web page.

Instructions:
- Select all AX Tree fragments (i.e., nodes and their subtrees) that are directly or indirectly relevant to the task or sub-task.
- Err on the side of inclusion: **if you are unsure whether an element is needed, include it.**
- You must **copy the selected fragments exactly as-is** from the original AX Tree.
- **Do not modify** any content — including the AXNode ID, node type, labels, values, or attributes.
- AXNode IDs (e.g., `[163]`) must **match the original AX Tree exactly**.
- You may extract multiple non-contiguous branches if needed.
- Capture as much of the environment as possible to avoid omissions or errors.

The extracted sub-environment must be formatted inside triple backticks.

---

### Output format:

```
[The extracted sub-environment]
```


# Current task:
Task:
{task}

# Current Page
Accessibility Tree:
{truncated_env}
        '''
        reasoning_content, response = self.llm(prompt)
        html_pattern = r"```(.*?)```"
        html_blocks = re.findall(html_pattern, response, re.DOTALL)
        html_blocks = "\n".join(html_blocks)
        expanded_text, m = expand_subenv_once_to_text_wide_with_metrics(
            original_text=html,
            subenv_text=html_blocks,
            mode="focused",  # "focused"|"balanced"|"wide"
        )
        html_blocks = [html_blocks, expanded_text, m]
        record = self._save(
            agent_name="EnvSegAgent",
            axtree=html,
            task=goal,
            subtask=task,
            input=prompt,
            output=response,
            extracted_part=html_blocks,
            reasoning_content=reasoning_content,
            step=self.counter
        )

        return html_blocks, record

    def _parser(self, response, pattern=r"```(.*?)```"):
        extracted_part = re.findall(pattern, response, re.DOTALL)
        if len(extracted_part) == 0:
            print('(Task decomposition) Format error.')
            raise ValueError(f"(Task decomposition)The response does not contain the expected format.")
        return extracted_part[0]

    def _save(self, agent_name, input, output, extracted_part, reasoning_content, axtree, task, subtask, step):
        record = AgentIO(
            step=step,
            agent_name=agent_name,
            task=task,
            subtask=subtask,
            axtree=axtree,
            input=input,
            output=output,
            extracted_part=extracted_part,
            reasoning_content=reasoning_content,
            timestamp=time.time()
        )

        return json.dumps(asdict(record), ensure_ascii=False)


    def _action_judge(self, action_intent, subtask, history, action, element, task_goal, subenv, env):

        prompt = f"""
You are an expert web automation evaluator, responsible for assessing both action execution success and the quality of task/environment decomposition made by an AI agent during a web task.
Your judgment combines two perspectives:
Action Execution Accuracy — Did the executed action actually achieve its intended purpose?
Decomposition Quality — Was the sub-task (and its supporting sub-environment) well-defined, logically placed, and consistent with the overall task and environment?
You will be provided with:
1. Overall Task Goal — the agent's high-level objective.
2. Overall Environment of Current Step — the full environment before sub-environment extraction.
3. Action History — a list of previously executed steps.
4. Executed Action — the specific action the agent just performed.
5. Detailed Element — the DOM element involved in the action.
6. Action Intent — the natural-language description of what the agent wanted to achieve with the action.
7. Sub-task — the decomposed step corresponding to this action.
8. Sub-environment — the extracted sub-environment supporting that sub-task.

Part 1: Action Execution Success
You must determine whether the executed action successfully fulfilled its intent.
- Based on the action and environment, determine whether the action can complete the subtask.
- If the action did not align with sub-task exactly but still contributes positively toward achieving the overall task goal, count it as successful.
- Provide a brief explanation of why the action succeeded or failed.


Output either:
true or false inside code fences for the result.

Part 2: Task and Environment Decomposition Quality

Sub-task Quality (0-5)
A high-quality sub-task should be atomic, precise, and logically aligned with both the task goal and previous steps.
Scoring reference:
5: Clear, atomic, non-redundant, consistent with the goal.
4: Slightly vague but mostly correct.
3: Somewhat unclear or contains mild future assumptions.
2: Poorly scoped or confusing.
1: Directionally wrong or ambiguous.
0: Not actionable or irrelevant.

Here are some examples for task quality judging:
#### Example 1: Multiple actions
subtask:
Click the tab labeled 'Customers' to access the customer list and search for the phone number 8015551212.
subtask_score: 2
subtask_reasoning: The sub-task combines two actions (clicking the tab and searching for the phone number), which makes it non-atomic. It also references a search that hasn't been executed yet, creating ambiguity.

#### Example 2: Single action
subtask: Navigate to the 'CUSTOMERS' section from the menubar.
subtask_score: 5
subtask_reasoning: The sub-task 'Navigate to the CUSTOMERS section' is atomic, directly corresponds to clicking the 'CUSTOMERS' menubar link (element 207), and logically precedes data retrieval. It aligns with the task goal of accessing customer details.

#### Example 3: Strong Future Assumption Example
Sub-task:
Click the 'March Orders' tab to calculate total spending for March 2022, assuming relevant orders are displayed.

subtask_score: 2
subtask_reasoning: The sub-task includes a speculative assumption about the page content after clicking, which may or may not be true.

#### Example 4: Mildly problematic (Weak Assumption) Example
Sub-task:
Click the link labeled 'Phone Cases' to identify EYZUTAK cases in the resulting list.

subtask_score: 3
subtask_reasoning: The second clause presumes the presence of a specific brand after the click, introducing mild future prediction.

#### Example 5: Good Example (Pure Action)
Sub-task:
Click the link labeled 'Phone Cases' to view available phone case products.

subtask_score: 5
subtask_reasoning: The sub-task describes a single, clearly defined action without assumptions about what will be found.
Sub-environment Quality (0-5)
A valid sub-environment must be a verbatim subset of the original environment and contain all elements necessary to perform the sub-task.
Scoring reference:
5: Fully correct, complete, and structurally consistent.
4: Minor omissions but still workable.
3: Noticeable omissions yet partially usable.
2: Missing key elements, difficult to use.
1: Nearly unusable.
0: Contains fabricated or altered elements.

Required Output Format (JSON)
{{
  "action_success": true or false,
  "action_reasoning": "why the action succeeded or failed",
  "subtask_score": int (0-5),
  "subenv_score": int (0-5),
  "subtask_reasoning": "1-2 sentences explaining the sub-task quality judgment",
  "subenv_reasoning": "1-2 sentences explaining the sub-environment quality judgment",
}}

Current Evaluate
Overall Task Goal:
{task_goal}

Overall Environment of Current Step:
{env}

Previous Steps:
{self.history.parent_history()}

Current Step Sub-task:
{subtask}

Current Step Sub-environment:
{subenv}

Executed Action:
{action}

Detailed Element:
{element}

Action Intent:
{action_intent}
        """
        res = truncate_prompt(
            base_prompt=prompt,
            history=self.history,
            environment=env,
            max_tokens=30000,
            reserve=30,
            model_name="utils/tokenizer",
            keep_last_n_history=3
        )
        if res.is_truncated:
            if res.truncated_hist:
                truncated_history = res.truncated_part["history"]
                truncated_history = truncated_history.strip().split("\n")[:-1]
                truncated_history = "\n".join(truncated_history) if truncated_history else "No history, this is your first step."
            else:
                truncated_history = self.history.parent_history()
            if res.truncated_env:
                truncated_env = res.truncated_part["env"]
            else:
                truncated_env = env
            prompt = f"""
You are an expert web automation evaluator, responsible for assessing both action execution success and the quality of task/environment decomposition made by an AI agent during a web task.
Your judgment combines two perspectives:
Action Execution Accuracy — Did the executed action actually achieve its intended purpose?
Decomposition Quality — Was the sub-task (and its supporting sub-environment) well-defined, logically placed, and consistent with the overall task and environment?
You will be provided with:
1. Overall Task Goal — the agent's high-level objective.
2. Overall Environment of Current Step — the full environment before sub-environment extraction.
3. Action History — a list of previously executed steps.
4. Executed Action — the specific action the agent just performed.
5. Detailed Element — the DOM element involved in the action.
6. Action Intent — the natural-language description of what the agent wanted to achieve with the action.
7. Sub-task — the decomposed step corresponding to this action.
8. Sub-environment — the extracted sub-environment supporting that sub-task.

Part 1: Action Execution Success
You must determine whether the executed action successfully fulfilled its intent.
- Based on the action and environment, determine whether the action can complete the subtask.
- If the action did not align with sub-task exactly but still contributes positively toward achieving the overall task goal, count it as successful.
- Provide a brief explanation of why the action succeeded or failed.
Output either:
true or false inside code fences for the result.

Part 2: Task and Environment Decomposition Quality

Sub-task Quality (0-5)
A high-quality sub-task should be atomic, precise, and logically aligned with both the task goal and previous steps.
Scoring reference:
5: Clear, atomic, non-redundant, consistent with the goal.
4: Slightly vague but mostly correct.
3: Somewhat unclear or contains mild future assumptions.
2: Poorly scoped or confusing.
1: Directionally wrong or ambiguous.
0: Not actionable or irrelevant.

Here are some examples for task quality judging:
#### Example 1: Multiple actions
subtask:
Click the tab labeled 'Customers' to access the customer list and search for the phone number 8015551212.
subtask_score: 2
subtask_reasoning: The sub-task combines two actions (clicking the tab and searching for the phone number), which makes it non-atomic. It also references a search that hasn't been executed yet, creating ambiguity.

#### Example 2: Single action
subtask: Navigate to the 'CUSTOMERS' section from the menubar.
subtask_score: 5
subtask_reasoning: The sub-task 'Navigate to the CUSTOMERS section' is atomic, directly corresponds to clicking the 'CUSTOMERS' menubar link (element 207), and logically precedes data retrieval. It aligns with the task goal of accessing customer details.

#### Example 3: Strong Future Assumption Example
Sub-task:
Click the 'March Orders' tab to calculate total spending for March 2022, assuming relevant orders are displayed.

subtask_score: 2
subtask_reasoning: The sub-task includes a speculative assumption about the page content after clicking, which may or may not be true.

#### Example 4: Mildly problematic (Weak Assumption) Example
Sub-task:
Click the link labeled 'Phone Cases' to identify EYZUTAK cases in the resulting list.

subtask_score: 3
subtask_reasoning: The second clause presumes the presence of a specific brand after the click, introducing mild future prediction.

#### Example 5: Good Example (Pure Action)
Sub-task:
Click the link labeled 'Phone Cases' to view available phone case products.

subtask_score: 5
subtask_reasoning: The sub-task describes a single, clearly defined action without assumptions about what will be found.
Sub-environment Quality (0-5)
A valid sub-environment must be a verbatim subset of the original environment and contain all elements necessary to perform the sub-task.
Scoring reference:
5: Fully correct, complete, and structurally consistent.
4: Minor omissions but still workable.
3: Noticeable omissions yet partially usable.
2: Missing key elements, difficult to use.
1: Nearly unusable.
0: Contains fabricated or altered elements.

Required Output Format (JSON)
{{
  "action_success": true or false,
  "action_reasoning": "why the action succeeded or failed",
  "subtask_score": int (0-5),
  "subenv_score": int (0-5),
  "subtask_reasoning": "1-2 sentences explaining the sub-task quality judgment",
  "subenv_reasoning": "1-2 sentences explaining the sub-environment quality judgment",
}}

Current Evaluate
Overall Task Goal:
{task_goal}

Overall Environment of Current Step:
{truncated_env}

Previous Steps:
{truncated_history}

Current Step Sub-task:
{subtask}

Current Step Sub-environment:
{subenv}

Executed Action:
{action}

Detailed Element:
{element}

Action Intent:
{action_intent}
        """
        result = False
        try_num = 0
        while not result and try_num < 5:
            reasoning_content, response = self.llm(prompt)
            result = self.extract_llm_json(response.strip(), prompt, agent_name='JudgeAgent')
            try_num += 1
        if not result:

            raise Exception("Failed to extract JSON from Judge Agent output")

        record = self._save(
            agent_name="ActionJudgeAgent",
            task=task_goal,
            subtask=subtask,
            axtree=env,
            input=prompt,
            output=response,
            extracted_part=result,
            reasoning_content=reasoning_content,
            step=self.counter
        )

        try:
            if result["action_success"] and int(result["subtask_score"]) >= 4 and int(result["subenv_score"]) >= 4:
                return True, response.strip(), record
            else:
                return False, response.strip(), record
        except Exception as e:
            raise e

    def summary_agent(self, task, action, intent, axtree, element):
        prompt = f"""
You are a Summarization Agent in a web automation system.

Your role is to produce a concise, accurate, and context-aware summary of the current step.
Your summary must reflect (1) the current page environment, (2) what element was manipulated, (3) the action’s intent, and also incorporate the judge result to reflect whether the step was meaningful or successful.

You are given:
1. **Task Goal**: The overall objective of the task.
2. **Overall Environment of Current Step**: The full environment of current step.
3. **Action**: The action to be executed.
4. **Manipulated Element**: The element that is manipulated by the action.
5. **Action Intent**: A description of what the action is trying to achieve.

Your summary must include:
1. **Page Environment Description**: Briefly describe the functional nature of the current page, such as:
   - "an online shopping page with a product search bar"
   - "a login page requiring username and password input"
Do NOT list specific DOM nodes. Summarize the environment as a key functional areas relevant to the action.

2. **Manipulated Element**: Describe which element was interacted with, in natural language.

3. **The Summary of the Action**: Explain what the action achieves based on the given information.

Do not include reasoning or explanation — just the summary.

Required Output Format (JSON)
{{
    "environment_description": "a brief description of the functional areas of the page",
    "manipulated_element": "the element that is manipulated by the action",
    "action_summary": "a description of what the action achieves",
}}

# Current task
Task Goal: {task}
Overall Environment of Current Step: {axtree}
Action: {action}
Manipulated Element: {element}
Action Intent: {intent}
        """
        res = truncate_prompt(
            base_prompt=prompt,
            history=None,
            environment=axtree,
            max_tokens=30000,
            reserve=30,
            model_name="utils/tokenizer",
            keep_last_n_history=3
        )
        if res.is_truncated:
            if res.truncated_hist:
                truncated_history = res.truncated_part["history"]
                truncated_history = truncated_history.strip().split("\n")[:-1]
                truncated_history = "\n".join(truncated_history) if truncated_history else "No history, this is your first step."
            else:
                truncated_history = self.history.parent_history()
            if res.truncated_env:
                truncated_env = res.truncated_part["env"]
            else:
                truncated_env = axtree
            prompt = f"""
You are a Summarization Agent in a web automation system.

Your role is to produce a concise, accurate, and context-aware summary of the current step.
Your summary must reflect (1) the current page environment, (2) what element was manipulated, (3) the action’s intent, and also incorporate the judge result to reflect whether the step was meaningful or successful.

You are given:
1. **Task Goal**: The overall objective of the task.
2. **Overall Environment of Current Step**: The full environment of current step.
3. **Action**: The action to be executed.
4. **Manipulated Element**: The element that is manipulated by the action.
5. **Action Intent**: A description of what the action is trying to achieve.

Your summary must include:
1. **Page Environment Description**: Briefly describe the functional nature of the current page, such as:
   - "an online shopping page with a product search bar"
   - "a login page requiring username and password input"
Do NOT list specific DOM nodes. Summarize the environment as a key functional areas relevant to the action.

2. **Manipulated Element**: Describe which element was interacted with, in natural language.

3. **The Summary of the Action**: Explain what the action achieves based on the given information.

Do not include reasoning or explanation — just the summary.

Required Output Format (JSON)
{{
    "environment_description": "a brief description of the functional areas of the page",
    "manipulated_element": "the element that is manipulated by the action",
    "action_summary": "a description of what the action achieves",
}}

# Current task
Task Goal: {task}
Overall Environment of Current Step: {truncated_env}
Action: {action}
Manipulated Element: {element}
Action Intent: {intent}
        """
        result = False
        try_num = 0
        while not result and try_num < 5:
            reasoning_content, response = self.llm(prompt)
            result = self.extract_llm_json(response.strip(), prompt, agent_name="SummaryAgent")
            try_num += 1
        if not result:
            raise Exception("Failed to extract JSON from Judge Agent output")

        record = self._save(
            agent_name="SummaryAgent",
            task=task,
            subtask=None,
            axtree=axtree,
            input=prompt,
            output=response,
            extracted_part=result,
            reasoning_content=reasoning_content,
            step=self.counter
        )

        return result, record

    def _false_message_agent(self, task, subtask, action, action_intent, judge_cot, axtree):
        prompt = f"""
You are a failure explanation agent for a web automation system.

Your job is to analyze why a given action failed to achieve its intended effect, and to summarize the issue clearly along with suggestions to avoid or fix it in the future.

You are given:

1. **Task Goal**: The overall goal the agent is trying to accomplish.
2. **Sub-task**: The current sub-goal the agent was performing.
3. **Action**: The action that was executed.
4. **Action Intent**: What the agent was trying to achieve with that action.
5. **Failure Analysis**: A reasoning description of why the action failed.

Your output should contain:
- A concise summary of what went wrong.
- If possible, the underlying cause.

**Output format:**

Failure Summary:
[Describe what went wrong.]

Cause Analysis:
[Explain the likely root cause.]

# Current task
Task Goal: {task}
Sub-task: {subtask}
Action: {action}
Action Intent: {action_intent}
Failure Analysis: {judge_cot}
        """
        reasoning_content, response = self.llm(prompt)
        false_message = f"""
Subtask: {subtask}
Action: {action}
Action intent:
{action_intent}
Analysis:
{response}
Conclusion: bad attempt.
        """
        record = self._save(
            agent_name="FalseMessageAgent",
            task=task,
            subtask=subtask,
            axtree=axtree,
            input=prompt,
            output=response,
            extracted_part=false_message,
            reasoning_content=reasoning_content,
            step=self.counter
        )
        return false_message, record

    def _is_subenv_match(self, original: str, subset: str) -> bool:
        """Check whether a sub-environment is a verbatim subset of the full AXTree.

        The check is performed line by line after trimming whitespace.

        Args:
            original: The full AXTree text.
            subset: The extracted sub-environment text.

        Returns:
            True if every non-empty line in ``subset`` appears in ``original``.
        """
        original_lines = set(line.strip() for line in original.splitlines() if line.strip())
        subset_lines = set(line.strip() for line in subset.splitlines() if line.strip())

        unmatched_lines = subset_lines - original_lines
        if unmatched_lines:
            print("Mismatch lines found:")
            for line in unmatched_lines:
                print(f"❌ Not found in original: {line}")
            return False
        else:
            print("Sub-environment match successfully")
        return True

    def extract_llm_json(self, output_text: str, prompt: str, agent_name: str):
        """Extract the first JSON object from an LLM response.

        Expected keys include ``action_success``, ``action_reasoning``,
        ``subtask_score``, ``subenv_score``, ``subtask_reasoning``, and
        ``subenv_reasoning``.
        """
        # Match the JSON object enclosed by curly braces.
        json_pattern = re.compile(r'\{[\s\S]*?\}')
        match = json_pattern.search(output_text)
        if not match:
            self.save_with_timestamp(f'No JSON object was found.\nAgent name: {agent_name}\n\nInput: {prompt}. \n\nOutput: {output_text}',
            self.error_log_path)
            raise ValueError("No JSON object was found in the LLM output.")

        json_str = match.group(0)

        # Normalize Python-style booleans to valid JSON booleans.
        json_str = json_str.replace("True", "true").replace("False", "false")

        # Parse the extracted string as JSON.
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Return False so the caller can retry on malformed JSON.
            print(f"Error JSON: {json_str}")
            return False

        return data

    def save_with_timestamp(
        self,
        content: str,
        directory: str,
        prefix: str = "log",
        time_format: str = "%Y%m%d_%H%M%S",
    ) -> str:
        """Write text content to a timestamped log file.

        Args:
            content: Text content to write.
            directory: Directory where the log file should be saved.
            prefix: Prefix of the generated file name.
            time_format: Datetime format used in the generated file name.

        Returns:
            The full path of the generated log file.
        """

        # Build the timestamped file name.
        timestamp = datetime.now().strftime(time_format)

        # Create the final log file name.
        filename = f"{prefix}_{timestamp}.txt"

        # Ensure the target directory exists.
        os.makedirs(directory, exist_ok=True)

        # Build the full output path.
        file_path = os.path.join(directory, filename)

        # Write the log content.
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
