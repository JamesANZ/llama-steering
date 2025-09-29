'''
Create train and eval datasets from alpaca_eval.json
'''
import json
import random

seed = 16
random.seed(seed)

with open("data/alpaca_eval.json", "r") as f:
    data = json.load(f)

prompts = [{"instruction" : d["instruction"]} for d in data]
random.shuffle(prompts)

# Split 50/50
split_idx = len(prompts) // 2
train_prompts = prompts[:split_idx]
eval_prompts = prompts[split_idx:]

# Save to files
with open("data/alpaca_train_prompts.json", "w") as f:
    json.dump(train_prompts, f, indent=2)
with open("data/alpaca_eval_prompts.json", "w") as f:
    json.dump(eval_prompts, f, indent=2)