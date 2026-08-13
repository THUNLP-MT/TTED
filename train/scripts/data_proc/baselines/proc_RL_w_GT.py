import json
import pandas as pd
import random
import os

def process_large_json(raw_data):
    processed_list = []
    
    task_ids_path = "train/src/task_ids/int_task_ids.json"
    with open(task_ids_path, 'r', encoding='utf-8') as f:
        task_ids = json.load(f)

    for entry in raw_data:
        task_id = entry["id"]
        if task_id not in task_ids:
            continue
        total_steps = len(entry["samples"])
        for step_idx, step in enumerate(entry["samples"]):
            for agent_info in step:
                if agent_info["agent_name"] != "ActionAgent":
                    continue
                item = {
                    "data_source": "webarena",
                    # "prompt": agent_info["input"],
                    # "response": agent_info["output"],
                    "messages": [
                        {"role": "user", "content": agent_info["input"]},
                        {"role": "assistant", "content": "<think>" + agent_info["reasoning_content"] + "</think>" + agent_info["output"]}
                    ],
                    "enable_thinking": True,
                    "reward": agent_info["label"],
                    "ability": "web_agent", 
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
