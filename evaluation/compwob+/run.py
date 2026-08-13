import argparse
import random
import re
import time

from openai import OpenAI
import os
import httpx
import computergym
import gym

import logging
from miniwob.miniwob_interface.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
from selenium.webdriver.common.keys import Keys

logging.basicConfig(level=logging.INFO)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument(
        "--num-episodes",
        "--num_episodes",
        dest="num_episodes",
        type=int,
        default=10,
    )
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--base_url", type=str, default="none")
    parser.add_argument("--api_key", type=str, required=True)
    opt = parser.parse_args()

    return opt


def llm(prompt, thinking_budget=5000, output_budget=10000, user_message = None):
    openai_api_key = opt.api_key
    openai_api_base = opt.base_url

    client = OpenAI(api_key=openai_api_key,
                    base_url=openai_api_base,
                    http_client=httpx.Client(trust_env=False))
    model = opt.llm
    if user_message is None:
        messages = [
            {"role": "user", "content": prompt}]
    else:
        messages = user_message
    response = client.chat.completions.create(model=model, messages=messages, temperature=0,max_tokens=thinking_budget, extra_body={"chat_template_kwargs": {"enable_thinking": True},})
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    if response.choices[0].finish_reason == "length":
        if reasoning_content is None:
            truncated_reasoning = content + '\n</think>'
        else:
            truncated_reasoning = f"<think>\n{reasoning_content}\n</think>"
        if user_message is None:
            new_prompt = prompt + '\n' + truncated_reasoning
            new_messages = [{"role": "user", "content": new_prompt}]
        else:
            user_prompt = user_message[1]["content"]
            new_prompt = user_prompt + '\n' + truncated_reasoning
            new_messages = [{"role": user_prompt[0]["role"], "content": user_prompt[0]["content"]},
                            {"role": user_prompt[1]["role"], "content": new_prompt}]
        new_response = client.chat.completions.create(model=model, messages=new_messages, temperature=0,
                                                        max_tokens=output_budget, extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                })
        new_content = new_response.choices[0].message.content
        new_reasoning_content = new_response.choices[0].message.reasoning_content

        return new_reasoning_content, new_content
    return reasoning_content, content

def miniwob(opt):
    # Step 1: Create the environment and initialize the agent.
    # Create a new MiniWoBEnv-v0 environment with gym.make().
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)
    # Create a new LLMAgent instance for each episode.
    success = 0
 
    with open(f"base_prompt/example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    with open(f"base_prompt/base.txt", "r", encoding="utf-8") as f:
        base = f.read()
    # Step 2: Run the interaction between the environment and the agent.
    false_record = []
    tasks = []
    htmls = []
    for round in range(opt.num_episodes):

        # Initialize the environment by resetting it with env.reset().
        seeds = [random.random()]
        states = env.reset(seeds=seeds, record_screenshots=True)
        task = states[0].utterance  # Set states[0].utterance as the agent's objective.
        # print(task)
        # Update the agent's HTML state.


        dones = [False]

        html_state = get_html_state(opt, states)  # Generate an HTML representation of the current state.
        prompt = f"{base}\n" \
                 f"Here are some examples for specific tasks, please follow the example to generate your answer for the current task.\n" \
                 f"\n" \
                 f"# Example\n" \
                 f"{example}\n" \
                 f"\n"
        subtask_prompt = prompt + f"# Current Task\n" \
                                  f"Below is the HTML code of the webpage where the agent should solve a task.\n{html_state}\n" \
                                  f"Current Task: {task}\n" \
                                  f"Directly output the instructions, and do not include any thought and explanatory comments in your response.\n"
        subtask_prompt += "Answer:\n"

        reasoning, response = llm(subtask_prompt)
        print(response)
        try:
            answer = response.split("\n")
            answer = extract_valid_instructions(answer)
            for ans in answer:
                ans = re.sub(r'^\d+\.\s*', '', ans)
                miniwob_action = convert_to_miniwob_action(
                    ans)  # Convert the generated instruction into an executable MiniWoB action.

                states, rewards, dones, _ = env.step([miniwob_action])
                # time.sleep(2)
        except ValueError:
            print("Invalid action")
            rewards = [0]
            dones = [True]
            rewards[0] = -1

        # A nonzero reward indicates that the episode has finished.
        if rewards[0] > 0 and dones == [True]:
            success += 1
        elif rewards[0] <= 0 and dones == [True]:
            false_record.append(response)
            tasks.append(task)
            htmls.append(html_state)
        print(f"success rate: {success} / {round + 1} = {success / (round + 1)}")

    record_dir = f"record/{opt.env}"
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    with open(f"{record_dir}/result_record.txt", "w", encoding="utf-8") as f:
        f.write(f"{opt.env}, {success}/{opt.num_episodes}={success / opt.num_episodes}\n")
    env.close()


def get_html_state(opt, states):
    extra_html_task = [
        "click-dialog",
        "click-dialog-2",
        "use-autocomplete",
        "choose-date",
    ]

    html_body = states[0].html_body
    for task in extra_html_task:
        if task in opt.env:
            html_body += states[0].html_extra
            break
    return html_body


def convert_to_miniwob_action(instruction: str):
    instruction = instruction.split(" ")
    inst_type = instruction[0]
    inst_type = inst_type.lower()

    if inst_type == "type":
        characters = " ".join(instruction[1:])
        characters = characters.replace('"', "")
        return MiniWoBType(characters)
    elif inst_type == "clickid":
        element_id = " ".join(instruction[1:])
        return MiniWoBElementClickId(element_id)
    elif inst_type == "press":
        key_type = instruction[1].lower()
        if key_type == "enter":
            return MiniWoBType("\n")
        elif key_type == "space":
            return MiniWoBType(" ")
        elif key_type == "arrowleft":
            return MiniWoBType(Keys.LEFT)
        elif key_type == "arrowright":
            return MiniWoBType(Keys.RIGHT)
        elif key_type == "backspace":
            return MiniWoBType(Keys.BACKSPACE)
        elif key_type == "arrowup":
            return MiniWoBType(Keys.UP)
        elif key_type == "arrowdown":
            return MiniWoBType(Keys.DOWN)
        else:
            raise NotImplementedError()
    elif inst_type == "movemouse":
        xpath = " ".join(instruction[1:])
        xpath = clean_xpath(xpath)
        return MiniWoBMoveXpath(xpath)
    elif inst_type == "clickxpath":
        xpath = " ".join(instruction[1:])
        xpath = clean_xpath(xpath)
        return MiniWoBElementClickXpath(xpath)
    elif inst_type == "clickoption":
        xpath = " ".join(instruction[1:])
        xpath = clean_xpath(xpath)
        return MiniWoBElementClickOption(xpath)
    else:
        raise ValueError("Invalid instruction")


def clean_xpath(xpath: str):
    if xpath[0] == '"':
        xpath = xpath[1:]
    if xpath[-1] == '"':
        xpath = xpath[:-1]

    return xpath
def extract_valid_instructions(answer):
    valid_instructions = []
    for line in answer:
        line = line.strip()
        # Remove numbering such as "1. " or "2. ".
        line = re.sub(r"^\d+\.\s*", "", line)
        # Remove surrounding backticks, if present.
        line = re.sub(r"^`|`$", "", line)
        # Keep only instructions that start with a supported action name.
        if line.startswith(("clickxpath", "clickid", "type", "press", "movemouse", "clickoption")):
            valid_instructions.append(line)
    return valid_instructions

if __name__ == "__main__":
    opt = parse_opt()  # Parse command-line arguments.
    miniwob(opt)  # Run the MiniWoB evaluation.
