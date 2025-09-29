import torch
from nnsight import LanguageModel
import numpy as np
from pathlib import Path
import time
import argparse
import yaml
import shutil
import json

from steering import generate_steered_answer, compute_metrics, load_saes
from steering import print_memory_usage
from print_utils import pretty_vec, RED, GREEN, YELLOW, GRAY, EOC
from optimize import noisy_blackbox_optimization

# ======================================================================================================================

# Parse config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default = "scripts/optimize/optimize_botorch.yaml", help="Path to the config file.")
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg['description'])

# Prepare logging
time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_folder = Path('./logs/' + time_stamp + "_botorch_optimize")
log_folder.mkdir(parents=True, exist_ok=True)
shutil.copy(args.config, log_folder / "config.yaml")
print(f"Log folder: {log_folder}, config file copied.", flush=True)

print_memory_usage()

# Load model and SAEs vectors
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
llm = LanguageModel(cfg['llm_name'], dispatch=True, device_map="auto")
print_memory_usage()
steering_components = load_saes(cfg,device)
print_memory_usage()

# Load prompts
with open(cfg['prompt_dataset'], 'r') as f:
    data = json.load(f)
prompts = [d["instruction"] for d in data]

np.random.seed(cfg["seed"])

# ----------------------------------------------------------------------------------------------------------------------

def objective(x, verbose = True, num_evals = 1):
    ''' Objective function to minimize. Input x is a reduced vector, so we scale it by layer index '''
    xs = np.zeros((len(steering_components)))
    for i in range(len(steering_components)):
        steering_components[i]['strength'] = x[i] * steering_components[i]['layer']
        xs[i] = steering_components[i]['strength']

    avg_cost = 0
    avg_concept = 0
    avg_instruction = 0
    avg_fluency = 0
    avg_log_prob = 0
    avg_rep3 = 0

    for eval_idx in range(num_evals):
        # Pick a random prompt
        prompt = prompts[np.random.randint(0, len(prompts))]

        chat = [{"role": "system", "content": cfg["system_prompt"]}]  if cfg["system_prompt"] is not None else []
        chat.append({"role": "user", "content": prompt})

        output = generate_steered_answer(llm, chat, steering_components, cfg['max_new_tokens'], cfg['temperature'],
                                         repetition_penalty = cfg['repetition_penalty'],
                                         steer_prompt = cfg['steer_prompt'], clamp_intensity = cfg['clamp_intensity'])

        metrics = compute_metrics(llm, output, instruction=prompt, concept=cfg["concept"],
                                  use_llm_evaluation=cfg['use_llm_evaluation'], )

        # Compute score that we want to minimize : harmonic mean and a small helper cost if one of the scores is zero
        sc, si, sf = metrics['concept'], metrics['instruction'], metrics['fluency']
        if sc*si*sf < 1e-6:
            helper_cost = abs(metrics['avg_log_prob'] - cfg['target_log_prob']) + cfg['rep3_weight'] * metrics['rep3']
            cost = 1 + 0.2 * helper_cost
        else:
            hm = 3.0 / (1.0/(sc + 1e-6) + 1.0/(si + 1e-6) + 1.0/(sf + 1e-6))
            cost = 1 - 0.5 * hm

        if verbose:

            print(f"{GRAY}x={pretty_vec(x)} -> strengths={pretty_vec(xs)}, eval {eval_idx + 1}/{num_evals}{EOC}")
            print(f"Cost={cost:.2f}, C={sc}, I={si}, F={sf}, log_prob={metrics['avg_log_prob']:.2f}, rep3={metrics['rep3']:.2f}")
            print(f"Prompt=", prompt.replace('\n', ' '))
            print(f"Output={output['answer']}{EOC}")
            print("-" * 80, flush=True)

        avg_cost += cost / num_evals
        avg_concept += sc / num_evals
        avg_instruction += si / num_evals
        avg_fluency += sf / num_evals
        avg_log_prob += metrics['avg_log_prob'] / num_evals
        avg_rep3 += metrics['rep3'] / num_evals

    color = GREEN if avg_cost < 0.26 else YELLOW if avg_cost < 0.51 else ""
    print(f"{color}x={pretty_vec(x)} -> strengths={pretty_vec(xs)}")
    print(f"AVG ** Cost={avg_cost:.2f}, C={avg_concept}, I={avg_instruction}, F={avg_fluency}, log_prob={avg_log_prob:.2f}, rep3={avg_rep3:.2f}{EOC}")

    cost_components = {'x':x, 'strengths': xs,
                       'concept': avg_concept, 'instruction': avg_instruction, 'fluency': avg_fluency,
                       'avg_log_prob': avg_log_prob, 'rep3': avg_rep3, 'num_evals': num_evals}
    print_memory_usage()

    return torch.tensor([avg_cost], dtype = torch.float64), cost_components


# Create partial function for objective with fixed verbosity and num_evals
objective_function = lambda x: objective(x, verbose = True, num_evals = cfg['num_evals_per_call'])

# Run optimization
bounds = torch.tensor([[0.0] * len(steering_components), [cfg['max_bound']] * len(steering_components)], dtype=torch.float64)

# Draw random initial points within the bounds.
# We know the good solutions are likely to be around norm 0.4-0.8, so we sample accordingly.
x_init = torch.zeros((cfg['num_initial_points'], len(steering_components)), dtype=torch.float64)
min_norm_init, max_norm_init = 0.4, 0.8
for i in range(cfg['num_initial_points']):
    rand_point = torch.rand(len(steering_components), dtype=torch.float64) * (cfg['max_bound'] - 0.0) + 0.0
    target_norm = np.random.uniform(min_norm_init, max_norm_init)
    x_init[i,:] = rand_point * (target_norm / torch.norm(rand_point))

print_memory_usage()
noisy_blackbox_optimization(objective_function,
                            bounds = bounds,
                            x_init = x_init,
                            num_iterations = cfg['num_iterations'],
                            num_sobol_samples = cfg['num_sobol_samples'],
                            num_restarts = cfg['num_restarts'],
                            raw_samples = cfg['raw_samples'],
                            num_samples_per_iteration = cfg['num_samples_per_iteration'],
                            resample_best_interval = cfg['resample_best_interval'],
                            log_folder = log_folder
                            )
