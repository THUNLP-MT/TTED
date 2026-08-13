import ast
import json
import re
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ACTION_SCHEMAS = {
    'noop': {
        'params': ['wait_ms'],
        'defaults': {'wait_ms': 1000}
    },
    'scroll': {
        'params': ['delta_x', 'delta_y'],
        'defaults': {}
    },
    'keyboard_press': {
        'params': ['key'],
        'defaults': {}
    },
    'click': {
        'params': ['bid', 'button', 'modifiers'],
        'defaults': {'button': 'left', 'modifiers': []}
    },
    'fill': {
        'params': ['bid', 'value'],
        'defaults': {}
    },
    'hover': {
        'params': ['bid'],
        'defaults': {}
    },
    'tab_focus': {
        'params': ['index'],
        'defaults': {}
    },
    'new_tab': {'params': [], 'defaults': {}},
    'go_back': {'params': [], 'defaults': {}},
    'go_forward': {'params': [], 'defaults': {}},
    'goto': {
        'params': ['url'],
        'defaults': {}
    },
    'tab_close': {'params': [], 'defaults': {}},
    'select_option': {
        'params': ['bid', 'options'],
        'defaults': {}
    },
    'send_msg_to_user': {
        'params': ['text'],
        'defaults': {}
    },
    'report_infeasible': {
        'params': ['reason'],
        'defaults': {}
    }
}

IGNORE_ARGS_ACTIONS = {'noop', 'report_infeasible'}
FUZZY_ACTIONS = {'fill', 'send_msg_to_user'}


@dataclass
class ActionObj:
    original_text: str
    func_name: str
    
    normalized_kwargs: Dict[str, Any]
    
    strict_key: str = ""      # 用于严格匹配
    fuzzy_text: Optional[str] = None # 用于 Embedding
    
    is_valid: bool = False
    embedding: Optional[np.ndarray] = None


class ActionNormalizer:
    @staticmethod
    def normalize(func_name: str, args: List[Any], keywords: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将位置参数和关键字参数映射到 Schema，并填充默认值。
        如果参数不匹配（如参数过多、缺少必填参），返回 None。
        """
        if func_name not in ACTION_SCHEMAS:
            return None
        
        schema = ACTION_SCHEMAS[func_name]
        param_names = schema['params']
        defaults = schema['defaults']
        
        final_args = {}
        
        if len(args) > len(param_names):
            return None # 提供的参数多于定义
        
        for i, arg_val in enumerate(args):
            param_name = param_names[i]
            final_args[param_name] = arg_val
            
        for k, v in keywords.items():
            if k in final_args:
                return None # 参数冲突（既在位置里给了，又在kw里给了）
            if k not in param_names:
                return None # 也就是这是一个未知参数
            final_args[k] = v
            
        for p in param_names:
            if p not in final_args:
                if p in defaults:
                    final_args[p] = defaults[p]
                else:
                    return None # 缺少必填参数
        
        for k, v in final_args.items():
            if isinstance(v, list):
                try:
                    final_args[k] = sorted(v)
                except:
                    pass
                    
        return final_args

class RobustActionEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

    def _parse_action(self, text: str) -> ActionObj:
        text = text.strip()
        obj = ActionObj(original_text=text, func_name="", normalized_kwargs={})
        
        try:
            tree = ast.parse(text, mode='eval')
            if not isinstance(tree.body, ast.Call): return obj
            call = tree.body
            
            if not isinstance(call.func, ast.Name): return obj
            func_name = call.func.id
            obj.func_name = func_name
            
            raw_args = []
            for arg in call.args:
                try:
                    raw_args.append(ast.literal_eval(arg))
                except:
                    raw_args.append(str(arg)) # Fallback
            
            raw_keywords = {}
            for k in call.keywords:
                try:
                    raw_keywords[k.arg] = ast.literal_eval(k.value)
                except:
                    raw_keywords[k.arg] = str(k.value)

            normalized_dict = ActionNormalizer.normalize(func_name, raw_args, raw_keywords)
            
            if normalized_dict is None:
                # 归一化失败（参数错误），标记无效
                return obj
            
            obj.normalized_kwargs = normalized_dict
            obj.is_valid = True
            
            # 1. Ignore Args 类
            if func_name in IGNORE_ARGS_ACTIONS:
                obj.strict_key = "IGNORE"
                
            # 2. Fill 类 (局部模糊)
            elif func_name == 'fill':
                # fill 的 bid 是 strict key，value 是 embedding text
                obj.strict_key = str(normalized_dict['bid'])
                obj.fuzzy_text = str(normalized_dict['value'])
                
            # 3. Send Msg 类 (全模糊)
            elif func_name == 'send_msg_to_user':
                obj.strict_key = "FUZZY_ROOT"
                obj.fuzzy_text = str(normalized_dict['text'])
                
            # 4. Strict 类 (Click, Scroll 等)
            else:
                obj.strict_key = json.dumps(normalized_dict, sort_keys=True)
                
        except Exception as e:
            pass
            
        return obj

    def find_majority(self, action_lines: List[str], similarity_threshold: float = 0.75):
        parsed_actions = [self._parse_action(line) for line in action_lines]
        valid_actions = [x for x in parsed_actions if x.is_valid]
        
        if not valid_actions:
            return None

        fuzzy_candidates = [x for x in valid_actions if x.fuzzy_text is not None]
        if fuzzy_candidates:
            texts = [x.fuzzy_text for x in fuzzy_candidates]
            embeddings = self.model.encode(texts)
            for i, x in enumerate(fuzzy_candidates):
                x.embedding = embeddings[i]

        clusters = []
        
        by_func = {}
        for x in valid_actions: by_func.setdefault(x.func_name, []).append(x)
        
        for func, group in by_func.items():
            if func in IGNORE_ARGS_ACTIONS:
                clusters.append(group)
            
            elif func not in FUZZY_ACTIONS:
                strict_groups = {}
                for x in group: strict_groups.setdefault(x.strict_key, []).append(x)
                clusters.extend(strict_groups.values())
            
            else:
                strict_subgroups = {}
                for x in group: strict_subgroups.setdefault(x.strict_key, []).append(x)
                
                for sub_group in strict_subgroups.values():
                    if len(sub_group) == 1:
                        clusters.append(sub_group)
                        continue
                    
                    sub_embeddings = np.array([x.embedding for x in sub_group])
                    sim_matrix = cosine_similarity(sub_embeddings)
                    temp_clusters = []
                    used_indices = set()
                    
                    for i in range(len(sub_group)):
                        if i in used_indices: continue
                        current_cluster = [i]
                        used_indices.add(i)
                        
                        for j in range(i+1, len(sub_group)):
                            if j in used_indices: continue
                            if sim_matrix[i][j] >= similarity_threshold:
                                current_cluster.append(j)
                                used_indices.add(j)
                        temp_clusters.append(current_cluster)
                    
                    for indices in temp_clusters:
                        clusters.append([sub_group[i] for i in indices])

        if not clusters: return None
        clusters.sort(key=len, reverse=True)
        majority = clusters[0]

        return {
            "majority_action": majority[0].original_text,
            "majority_count": len(majority),
            "normalized_repr": majority[0].normalized_kwargs, # 展示归一化后的数据
            "instances": [x.original_text for x in majority]
        }


def run_non_decompose():
    engine = RobustActionEngine(model_name='all-MiniLM-L12-v2')
    
    task_ids_path = "train/src/task_ids/int_task_ids.json"
    with open(task_ids_path, 'r', encoding='utf-8') as f:
        task_ids = json.load(f)
    input_json_path = ""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data_list = []
    valid_item_steps = 0
    total_item_steps = 0
    for entry in tqdm(raw_data):
        task_id = entry["id"]
        if task_id not in task_ids:
            continue
        total_steps = len(entry["samples"])
        for step_idx, step in enumerate(entry["samples"]):

            action_list = []
            temp_item_list = []
            total_item_steps += 1
            for sampling_idx, agent_info in enumerate(step):
                if agent_info["agent_name"] != "Sampling":
                    continue
                
                action_text = agent_info["output"].split("### Action:")[-1].split("### Action intent:")[0].strip()
                item = {
                    "data_source": "webarena",
                    "id": f"{task_id}_{step_idx}_{sampling_idx}",
                    # "prompt": agent_info["input"],
                    # "response": agent_info["output"],
                    "messages": [
                        {"role": "user", "content": agent_info["input"]},
                        {"role": "assistant", "content": "<think>" + agent_info["reasoning_content"] + "</think>" + agent_info["output"]}
                    ],
                    "enable_thinking": True,
                    "ability": "web_agent", 
                    "extra_info": {
                        "task_id": task_id,
                        "task": agent_info["task"], 
                        "total_steps": total_steps,
                        "step_idx": step_idx,
                        "action_text": action_text,
                        **(agent_info["extracted_part"] if "extracted_part" in agent_info else {})
                    }
                }
                action_list.append(action_text)
                temp_item_list.append(item)

            result = engine.find_majority(action_list)
            if not result or len(result["instances"]) < 2:
                continue
            
            if len(result["instances"]) == len(action_list):
                continue

            total_reward_value = 0.0
            reward_value_list = []
            for item in temp_item_list:
                if item["extra_info"]["action_text"] in result["instances"]:
                    item["reward"] = 1.0
                else:
                    item["reward"] = 0.0
                total_reward_value += item["reward"]
                reward_value_list.append(item["reward"])
            # normalize reward
            for item in temp_item_list:
                item["reward"] = (item["reward"] - total_reward_value / len(temp_item_list)) / (np.std(reward_value_list) + 1e-5)
                processed_data_list.append(item)

            valid_item_steps += 1
                
    print("===============================")
    print(f"Total Steps: {total_item_steps}")
    print(f"Valid Steps: {valid_item_steps}")
    print(f"Valid Items: {len(processed_data_list)}")

    with open("", 'w', encoding='utf-8') as f:
        for item in processed_data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def parse_custom_json(json_input: str) -> dict:
    """
    将 JSON 字符串（支持被 ```json 包围的格式）转换为字典。
    """
    if not json_input:
        return None

    # 1. 去掉首尾多余的空格
    content = json_input.strip()

    # 2. 使用正则提取 ```json ... ``` 或 ``` ... ``` 之间的内容
    # 如果没有 markdown 标签，则保持原样
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        content = match.group(1)

    # 3. 解析 JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        # 如果解析失败，这里可以返回空字典或抛出异常，视你的业务需求而定
        return None
    
def run_decompose():
    engine = RobustActionEngine(model_name='all-MiniLM-L12-v2')
    
    task_ids_path = "train/src/task_ids/int_task_ids.json"
    with open(task_ids_path, 'r', encoding='utf-8') as f:
        task_ids = json.load(f)
    input_json_path = ""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data_list = []
    valid_item_steps = 0
    total_item_steps = 0
    for entry in tqdm(raw_data):
        task_id = entry["id"]
        if task_id not in task_ids:
            continue
        total_steps = len(entry["samples"])
        for step_idx, step in enumerate(entry["samples"]):

            action_list = []
            temp_item_list = []
            total_item_steps += 1
            accepted_action_agent_info, judge_res = None, None
            for sampling_idx, agent_info in enumerate(step):
                if agent_info["agent_name"] != "Sampling":
                    if agent_info["agent_name"] == "ActionAgent":
                        accepted_action_agent_info = agent_info
                    if agent_info["agent_name"] == "ActionJudgeAgent":
                        judge_res = parse_custom_json(agent_info["output"])
                    continue
                
                action_text = agent_info["output"].split("### Action:")[-1].split("### Action intent:")[0].strip()
                item = {
                    "data_source": "webarena",
                    "id": f"{task_id}_{step_idx}_{sampling_idx}",
                    # "prompt": agent_info["input"],
                    # "response": agent_info["output"],
                    "messages": [
                        {"role": "user", "content": agent_info["input"]},
                        {"role": "assistant", "content": "<think>" + agent_info["reasoning_content"] + "</think>" + agent_info["output"]}
                    ],
                    "enable_thinking": True,
                    "ability": "web_agent", 
                    "extra_info": {
                        "task_id": task_id,
                        "task": agent_info["task"], 
                        "total_steps": total_steps,
                        "step_idx": step_idx,
                        "action_text": action_text,
                        **(agent_info["extracted_part"] if "extracted_part" in agent_info else {})
                    }
                }
                action_list.append(action_text)
                temp_item_list.append(item)

            result = engine.find_majority(action_list)
            if not result or len(result["instances"]) < 2 or len(result["instances"]) == len(action_list):
                if judge_res is not None and "action_success" in judge_res:
                    accepted_action_text = accepted_action_agent_info["output"].split("### Action:")[-1].split("### Action intent:")[0].strip()
                    parsed_action = engine._parse_action(accepted_action_text)
                    if not parsed_action.is_valid:
                        print(f"Invalid action at task {task_id}, step {step_idx}: {accepted_action_text}")
                    item = {
                        "data_source": "webarena",
                        "id": f"{task_id}_{step_idx}_action",
                        # "prompt": accepted_action_agent_info["input"],
                        # "response": accepted_action_agent_info["output"],
                        "messages": [
                            {"role": "user", "content": agent_info["input"]},
                            {"role": "assistant", "content": "<think>" + agent_info["reasoning_content"] + "</think>" + agent_info["output"]}
                        ],
                        "enable_thinking": True,
                        "ability": "web_agent", 
                        "reward": 1.0 if judge_res["action_success"] and parsed_action.is_valid else 0.0,
                        "extra_info": {
                            "task_id": task_id,
                            "task": accepted_action_agent_info["task"], 
                            "total_steps": total_steps,
                            "step_idx": step_idx,
                            "action_text": accepted_action_text
                        }
                    }
                    processed_data_list.append(item)
                continue

            total_reward_value = 0.0
            reward_value_list = []
            for item in temp_item_list:
                if item["extra_info"]["action_text"] in result["instances"]:
                    item["reward"] = 1.0
                else:
                    item["reward"] = 0.0
                total_reward_value += item["reward"]
                reward_value_list.append(item["reward"])

            for item in temp_item_list:
                item["reward"] = (item["reward"] - total_reward_value / len(temp_item_list)) / (np.std(reward_value_list) + 1e-5)
                processed_data_list.append(item)

            valid_item_steps += 1
                
    print("===============================")
    print(f"Total Steps: {total_item_steps}")
    print(f"Valid Steps: {valid_item_steps}")
    print(f"Valid Items: {len(processed_data_list)}")

    with open("", 'w', encoding='utf-8') as f:
        for item in processed_data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_non_decompose()
