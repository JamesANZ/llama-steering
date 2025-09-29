import time
import numpy as np
from pathlib import Path
import json
import argparse
import yaml
import shutil
import torch
from nnsight import LanguageModel

from steering import generate_steered_answer, compute_metrics, load_saes


# ======================================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default = "scripts/sweep_1D/sweep_1D.yaml", help="Path to the config file.")
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg['description'])

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load model and SAEs
llm = LanguageModel(cfg['llm_name'], dispatch=True).to(device)
steering_components = load_saes(cfg, device)

# Load prompts
with open(cfg['prompt_dataset'], 'r') as f:
    data = json.load(f)
prompts = [d["instruction"] for d in data]

# Shuffle prompts and keep only num_prompts
np.random.seed(cfg['seed'])
np.random.shuffle(prompts)
prompts = prompts[:cfg['num_prompts']]


# ----------------------------------------------------------------------------------------------------------------------

time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_folder = Path('./logs/' + time_stamp + "_sweep_1D")
log_folder.mkdir(parents=True, exist_ok=True)
print(f"Log folder: {log_folder}")
shutil.copy(args.config, log_folder / "config.yaml")

results = []
total_num_steps = cfg['num_values'] * cfg['num_prompts']
start = time.time()
cnt_steps = 0

for steering_component in steering_components:
    print("Steering component:", steering_component)
    layer_idx, feature_idx = steering_component['layer'], steering_component['feature']

    # Compute step to nearest 0.1 to avoid float numbers with many decimals
    step = layer_idx * (cfg['max_intensity_reduced'] - cfg['min_intensity_reduced']) / (cfg['num_values'] - 1)
    step = round(step * 10) / 10
    steering_intensities = np.arange(layer_idx * cfg['min_intensity_reduced'], layer_idx * cfg['max_intensity_reduced'] + step, step)

    torch.manual_seed(cfg['seed'])
    for steering_intensity in steering_intensities:
        print("Steering intensity:", steering_intensity)
        steering_component['strength'] = steering_intensity
        torch.manual_seed(cfg['seed'])

        for prompt in prompts:
            chat = []
            if cfg["system_prompt"] is not None:
                chat = [{"role": "system", "content": cfg["system_prompt"]}]
            chat.append({"role": "user", "content": prompt})

            output = generate_steered_answer(llm, chat, [steering_component],
                                             max_new_tokens=cfg['max_new_tokens'],
                                             temperature=cfg['temperature'], repetition_penalty = cfg['repetition_penalty'],
                                             steer_prompt=cfg['steer_prompt'], clamp_intensity = cfg['clamp_intensity'])

            metrics = compute_metrics(llm, output, instruction = prompt, concept = cfg["concept"],
                                      use_llm_evaluation=cfg['use_llm_evaluation'], )

            results.append({
                "layer": layer_idx,
                "feature_index": feature_idx,
                "steering_intensity": steering_intensity,
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

            print(prompt)
            print(f"Log-prob: {metrics['avg_log_prob']:.2f}, rep3: {metrics['rep3']:.2f}, rep4: {metrics['rep4']:.2f}")
            if cfg['use_llm_evaluation']:
                print(f"LLM eval - concept: {metrics['concept']}, instruction: {metrics['instruction']}, fluency: {metrics['fluency']}")
            print(output['answer'])
            print(f"{cnt_steps}/{total_num_steps} ETA {finish_time}", end = " ")
            # print_memory_usage()
            print("-" * 80)

            # Save results
            if cnt_steps % 10 == 0 or cnt_steps == total_num_steps:
                with open(log_folder / "results.json", "w") as f:
                    json.dump(results, f, indent=2)