import json
import random
import os

original_path = ""
target_path = ""

with open(original_path, 'r', encoding='utf-8') as f:
    all_items = [json.loads(line) for line in f.readlines()]

seed = 42
train_ratio = 0.9

random.seed(seed)
random.shuffle(all_items)
split_idx = int(len(all_items) * train_ratio)
train_data = all_items[:split_idx]
val_data = all_items[split_idx:]

os.makedirs(target_path, exist_ok=True)

train_path = os.path.join(target_path, "train.jsonl")
val_path = os.path.join(target_path, "test.jsonl")

with open(train_path, 'w', encoding='utf-8') as f_train:
    for item in train_data:
        f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
with open(val_path, 'w', encoding='utf-8') as f_val:
    for item in val_data:
        f_val.write(json.dumps(item, ensure_ascii=False) + '\n')
