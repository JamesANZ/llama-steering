import json
import torch
from nnsight import LanguageModel
import argparse
import yaml
import time
import shutil
from pathlib import Path
from steering import load_saes, generate_steered_answer, compute_metrics
from print_utils import CLEAR_TERMINAL, EOC, RED, GREEN, YELLOW

# ----------------------------------------------------------------------------------------------------------------------

# Load config from YAML file
print("Parsing arguments...", flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default = "scripts/evaluation/evaluation.yaml", help="Path to the config file.")
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Prepare log folder
time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_folder = Path('./logs/' + time_stamp + "_evaluation")
log_folder.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config, log_folder / "config.yaml")
print(f"Log folder: {log_folder}, config.yaml saved.", flush=True)

# Load model & SAEs
print("Loading model...", flush=True)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
llm = LanguageModel(cfg['llm_name'], dispatch=True).to(device)
llm.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True     # suppress warning
print("Loading steering components...", flush=True)
steering_components = load_saes(cfg, device)

# Load prompts
with open(cfg['prompt_dataset'], 'r') as f:
    data = json.load(f)
prompts = [d["instruction"] for d in data]

# Main loop
results = []
total_num_steps = len(prompts)
start = time.time()
cnt_steps = 0
torch.manual_seed(cfg['seed'])
for prompt in prompts:

    chat = []
    if cfg["system_prompt"] is not None:
        chat = [{"role": "system", "content": cfg["system_prompt"]}]
    chat.append({"role": "user", "content": prompt})

    output = generate_steered_answer(llm, chat, steering_components,
                                     max_new_tokens=cfg['max_new_tokens'],
                                     temperature=cfg['temperature'], repetition_penalty = cfg['repetition_penalty'],
                                     steer_prompt=cfg['steer_prompt'], clamp_intensity = cfg['clamp_intensity'])
    metrics = compute_metrics(llm, output, instruction = prompt, concept = cfg["concept"],
                              use_llm_evaluation=cfg['use_llm_evaluation'], )

    results.append({
        "prompt": prompt,
        "avg_log_prob": metrics['avg_log_prob'],
        "greedy_avg_log_prob": metrics['greedy_avg_log_prob'],
        "rep3": metrics['rep3'],
        "rep4": metrics['rep4'],
        "llm_score_concept": metrics['concept'] if cfg['use_llm_evaluation'] else None,
        "llm_score_instruction": metrics['instruction'] if cfg['use_llm_evaluation'] else None,
        "llm_score_fluency": metrics['fluency'] if cfg['use_llm_evaluation'] else None,
        "answer": output['answer'],
    })

    cnt_steps += 1
    eta = (time.time() - start) / cnt_steps * (total_num_steps - cnt_steps)
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + eta))
    total_llm_score = metrics['concept'] + metrics['instruction'] + metrics['fluency']

    color = GREEN if total_llm_score > 4 else YELLOW if total_llm_score >= 3 else ""
    print("Prompt=", prompt.replace('\n', ' '))
    print(f"Output={output['answer']}")
    print(f"Score={total_llm_score:.2f}, C={metrics['concept']}, I={metrics['instruction']}, F={metrics['fluency']}, log_prob={metrics['avg_log_prob']:.2f}, rep3={metrics['rep3']:.2f}{EOC}")
    print(f"{cnt_steps}/{total_num_steps} ETA {finish_time}", end = " ")
    print("-" * 80)

    # Save results
    if cnt_steps % 10 == 0 or cnt_steps == total_num_steps:
        with open(log_folder / "results.json", "w") as f:
            json.dump(results, f, indent=2)