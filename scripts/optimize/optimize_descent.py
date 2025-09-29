import torch
import numpy as np
import math
from pathlib import Path
import argparse
import yaml

from optimize import gradient_descent_on_gp, load_gp
from print_utils import pretty_vec

# Load config from YAML file
print("Parsing arguments...", flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default = "scripts/optimize/optimize_descent.yaml", help="Path to the config file.")
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Load GP Model
print("Loading GP model...", flush=True)
checkpoint = torch.load(Path(cfg['folder']) / cfg["gp_file"] , weights_only=False)
gp, train_X, train_Y = load_gp(checkpoint)
gp.eval()
num_features = len(cfg["features"])
if num_features != train_X.shape[1]:
    print(f"ERROR: Mismatch in number of features! Config has {num_features}, GP was trained with {train_X.shape[1]}")
    exit(1)
best_location_list = checkpoint['best_location_list']
print(f"GP loaded: {train_X.shape[0]} observations, {num_features} features {len(best_location_list)} locations")


# Main Loop

print(f"\nRunning {cfg['num_restarts']} random restarts with beta={cfg['beta']}...")
print(f"Discarding solutions within {cfg['boundary_tolerance']} of boundaries")
print("=" * 80)

bounds_lower = torch.zeros(num_features, dtype=torch.float64)
bounds_upper = torch.ones(num_features, dtype=torch.float64) * cfg["upper_bound_factor"]
results = []
boundary_count = 0

for i in range(cfg['num_restarts']):
    # Random initialization from the best location list + jitter
    x_init = torch.tensor(best_location_list[np.random.randint(len(best_location_list))])
    x_init += cfg['jitter_coefficient'] * torch.randn(num_features, dtype=torch.float64)

    x_final, final_loss, steps, hit_upper_boundary = gradient_descent_on_gp(
        x_init, gp, cfg['beta'], cfg['lr'], cfg['max_steps'], float(cfg['tolerance']), bounds_lower, bounds_upper, float(cfg['boundary_tolerance']),
    )

    if hit_upper_boundary:
        boundary_count += 1
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{cfg['num_restarts']} restarts... ({boundary_count} hit boundary)")
        continue  # Skip this solution!

    # Get final statistics
    with torch.no_grad():
        posterior = gp.posterior(x_final.unsqueeze(0))
        final_mean = posterior.mean.item()
        final_std = posterior.variance.sqrt().item()
        predicted_objective = -final_mean

    results.append({
        'x': x_final.clone(),
        'objective': predicted_objective,
        'gp_mean': final_mean,
        'gp_std': final_std,
        'loss': final_loss,
        'steps': steps
    })

    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{cfg['num_restarts']} restarts... ({boundary_count} hit boundary)")


# Sort and Display Results
print("\n" + "=" * 80)
print(f"SUMMARY: {len(results)} interior minima found, {boundary_count} boundary solutions discarded")
print("=" * 80)

results.sort(key=lambda r: r['objective'])

print(f"\nTOP 10 INTERIOR MINIMA (beta={cfg['beta']}):")
print("=" * 80)

for i, r in enumerate(results[:10]):
    print(f"\n#{i + 1}: Predicted Objective = {r['objective']:.6f}")
    print(f"  Location: {pretty_vec(r['x'])}")
    print(f"  GP Mean: {r['gp_mean']:.4f}, GP Std: {r['gp_std']:.4f}")
    print(f"  Converged in {r['steps']} steps")



# Cluster to find unique minima
# =============================

print("\n" + "=" * 80)
print("CLUSTERING RESULTS (removing duplicates)...")

unique_minima = []
unique_minima_counts = []  # Track the size of each cluster
CLUSTER_THRESHOLD = 0.05 * math.sqrt(num_features)  # Distance threshold for clustering

for r in results:
    is_duplicate = False
    for idx, u in enumerate(unique_minima):
        dist = torch.norm(r['x'] - u['x']).item()
        if dist < CLUSTER_THRESHOLD:
            is_duplicate = True
            if r['objective'] < u['objective']:
                unique_minima[idx] = r
            unique_minima_counts[idx] += 1
            break

    if not is_duplicate:
        unique_minima.append(r)
        unique_minima_counts.append(1)

unique_minima_with_counts = list(zip(unique_minima, unique_minima_counts))
unique_minima_with_counts.sort(key=lambda rc: rc[0]['objective'])

MAX_DISPLAY = 30
print(f"Found {len(unique_minima)} unique interior minima:")
for i, (r, count) in enumerate(unique_minima_with_counts):

    rs = np.zeros(num_features)

    print(f"\n#{i + 1}: Objective = {r['objective']:.6f} (cluster size: {count})")
    print(f"  Location: {pretty_vec(r['x'], 3)}")
    print(f"  Uncertainty (σ): {r['gp_std']:.4f}")
    if i + 1 >= MAX_DISPLAY:
        print(f"\n... (only displaying top {MAX_DISPLAY} unique minima)")
        break

# Save Results
output_file = Path(cfg['folder']) / f"{Path(cfg['gp_file']).stem}_beta{cfg['beta']}_interior.pt"
torch.save({
    'all_results': results,
    'unique_minima': unique_minima,
    'beta': cfg['beta'],
    'num_restarts': cfg['num_restarts'],
    'boundary_count': boundary_count
}, output_file)

print(f"\nResults saved to: {output_file}")