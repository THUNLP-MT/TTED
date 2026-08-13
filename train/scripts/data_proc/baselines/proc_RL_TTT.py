import json
import random
import os
from typing import List, Dict, Any, Optional
import numpy as np
import ast
from dataclasses import dataclass

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
        
        # 1. 处理位置参数
        if len(args) > len(param_names):
            return None # 提供的参数多于定义
        
        for i, arg_val in enumerate(args):
            param_name = param_names[i]
            final_args[param_name] = arg_val
            
        # 2. 处理关键字参数
        for k, v in keywords.items():
            if k in final_args:
                return None
            if k not in param_names:
                return None
            final_args[k] = v
            
        # 3. 填充默认值 & 检查缺失
        for p in param_names:
            if p not in final_args:
                if p in defaults:
                    final_args[p] = defaults[p]
                else:
                    return None # 缺少必填参数
        
        # 4. 特殊处理：列表类型参数排序
        for k, v in final_args.items():
            if isinstance(v, list):
                try:
                    final_args[k] = sorted(v)
                except:
                    pass
                    
        return final_args

def parse_action(text: str) -> ActionObj:
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
            return obj
        
        obj.normalized_kwargs = normalized_dict
        obj.is_valid = True
        
        if func_name in IGNORE_ARGS_ACTIONS:
            obj.strict_key = "IGNORE"
            
        elif func_name == 'fill':
            obj.strict_key = str(normalized_dict['bid'])
            obj.fuzzy_text = str(normalized_dict['value'])
            
        elif func_name == 'send_msg_to_user':
            obj.strict_key = "FUZZY_ROOT"
            obj.fuzzy_text = str(normalized_dict['text'])
            
        else:
            obj.strict_key = json.dumps(normalized_dict, sort_keys=True)
            
    except Exception as e:
        pass
        
    return obj

def process_large_json(raw_data):
    """
    (2) JSON处理逻辑留空
    raw_data: 整个 JSON 文件加载后的内容（通常是一个大字典或大列表）
    返回：由多个 dict 组成的 list，每个 dict 符合你的格式要求
    """
    task_ids_path = "train/src/task_ids/int_task_ids.json"
    with open(task_ids_path, 'r', encoding='utf-8') as f:
        task_ids = json.load(f)
    processed_list = []
    
    for entry in raw_data:
        task_id = entry["id"]
        if task_id not in task_ids:
            continue
        total_steps = len(entry["samples"])
        for step_idx, step in enumerate(entry["samples"]):
            scores = {}
            item = None
            for agent_info in step:
                if agent_info["agent_name"] == "ActionJudgeAgent":
                    scores["action_score"] = 1 if agent_info["extracted_part"]["action_success"] else 0
                    if item is not None and item["reward"] is None:
                        item["reward"] = scores["action_score"]
                    continue
                if agent_info["agent_name"] not in ["ActionAgent"]:
                    assert agent_info["agent_name"] in ["SummaryAgent", "FalseMessageAgent"], f"Unexpected agent: {agent_info['agent_name']}"
                    continue
                parsed_action = parse_action(agent_info["extracted_part"]["action"])
                if not parsed_action.is_valid:
                    print(f"Invalid action at task {task_id}, step {step_idx}: {agent_info['extracted_part']['action']}")
                                
                item = {
                    "data_source": "webarena",
                    "id": f"{task_id}_{step_idx}",
                    # "prompt": agent_info["input"],
                    # "response": agent_info["output"],
                    "messages": [
                        {"role": "user", "content": agent_info["input"]},
                        {"role": "assistant", "content": "<think>" + agent_info["reasoning_content"] + "</think>" + agent_info["output"]}
                    ],
                    "enable_thinking": True,
                    "ability": "web_agent", 
                    "reward": scores.get("action_score", None) if parsed_action.is_valid else 0,
                    "extra_info": {
                        "task_id": task_id,
                        "task": agent_info["task"], 
                        "total_steps": total_steps,
                        "step_idx": step_idx,
                        **agent_info["extracted_part"]
                    }
                }
            processed_list.append(item)
    
    return processed_list

def convert_single_json_to_parquet(input_json_path, output_dir, train_ratio=0.9, seed=42):
    if not os.path.exists(input_json_path):
        print(f"Error: 文件 {input_json_path} 不存在")
        return

    print(f"正在读取文件: {input_json_path} ...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except Exception as e:
            print(f"解析 JSON 失败: {e}")
            return

    print("正在执行解析逻辑...")
    all_items = process_large_json(raw_data)
    
    if not all_items:
        print("未提取到任何有效数据，请检查 process_large_json 中的逻辑。")
        return

    print(f"提取完成，共 {len(all_items)} 条数据。正在洗牌...")
    random.seed(seed)
    random.shuffle(all_items)

    split_idx = int(len(all_items) * train_ratio)
    train_data = all_items[:split_idx]
    val_data = all_items[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "test.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f_train:
        for item in train_data:
            f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(val_path, 'w', encoding='utf-8') as f_val:
        for item in val_data:
            f_val.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"--- 转换成功 ---")
    print(f"训练集: {len(train_data)} 条 -> {train_path}")
    print(f"验证集: {len(val_data)} 条 -> {val_path}")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_JSON = ""
    OUTPUT_FOLDER = ""
    
    convert_single_json_to_parquet(INPUT_JSON, OUTPUT_FOLDER)
