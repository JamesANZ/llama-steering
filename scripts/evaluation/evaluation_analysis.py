import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml

SAVE_FIGS = True

folders = {
    'Prompt': "results/evaluation/20251017-150244_evaluation",
    'Basic steering': "results/evaluation/20251018-165901_evaluation",
    'Clamping': "results/evaluation/20251018-170431_evaluation",
    'Clamping + Penalty': "results/evaluation/20251019-105107_evaluation",
    '2D optimized' : "results/evaluation/20251023-160401_evaluation",
    '8D optimized' : "results/evaluation/20251025-052005_evaluation",
}

run_colors = [
              (0.3,0.3,0.3),
              (0.7,0.7,0.7),
              (0.7,0.7,0.8),
              (0.7,0.7,0.9),
              (0.7,1.0,0.7),
              (1.0,0.7,1.0),
]

metrics = [
    ("llm_score_concept", "LLM concept score", (0.0, 2)),
    ("eiffel", "Explicit concept presence", (0, 1)),
    ("llm_score_instruction", "LLM instruction score", (0.0, 2)),
    ("minus_log_prob", "Surprise in original model", (0.0, 2.0)),
    ("llm_score_fluency", "LLM fluency score", (0.0, 2)),
    ("rep3", "3-gram repetition fraction", (0.0, 1)),
    ("mean_llm_score", "Mean LLM score", (0.0, 2.0)),
    ("harmonic_llm_score", "Harmonic mean LLM score", (0.0, 2.0)),
]

# ======================================================================================================================

all_results = {}
all_stats = {}
for experiment in folders:

    # Read config.yaml file and print description
    with open(f"{folders[experiment]}/config.yaml", 'r') as file:
        cfg = yaml.safe_load(file)
        print(f"{experiment} : {cfg['description']}")

    # Load results and compute additional metrics
    results = pd.read_json(f"{folders[experiment]}/results.json")
    results['eiffel'] = results['answer'].str.lower().str.contains('eiffel').astype(int)
    results['minus_log_prob'] = - results['avg_log_prob']
    results['surprise'] = np.exp(- results['avg_log_prob'])
    results['mean_llm_score'] = (results['llm_score_concept'] + results['llm_score_instruction'] + results['llm_score_fluency']) / 3.0
    # Following AxBench, compute the harmonic mean of the three llm scores
    results['harmonic_llm_score'] = 3 / (
            1 / (results['llm_score_concept']) + 1 / (results['llm_score_instruction']) + 1 / (results['llm_score_fluency']))

    all_results [experiment] = results

# ======================================================================================================================

# Compute statistics for all metrics
for experiment, results in all_results.items():
    all_stats[experiment] = {}
    for im, metric in enumerate(metrics):
        metric_name = metric[0]
        if metric_name not in results:
            continue
        all_stats[experiment][metric_name] = {
            "mean": results[metric_name].mean(),
            "std": results[metric_name].std(),
            "median": results[metric_name].median(),
            "p10": results[metric_name].quantile(0.1),
            "p90": results[metric_name].quantile(0.9),
        }

    # extra : frequencies of each result for the harmonic mean
    all_stats[experiment]['harmonic_llm_score'].update({
        "freq_0": (results['harmonic_llm_score'] == 0.0).sum() / len(results),
        "freq_111": (results['harmonic_llm_score'] == 1.0).sum() / len(results),
        "freq_211": (results['harmonic_llm_score'] == 1.2).sum() / len(results),
        "freq_221": (results['harmonic_llm_score'] == 1.5).sum() / len(results),
        "freq_222": (results['harmonic_llm_score'] == 2.0).sum() / len(results),
    })

# ======================================================================================================================

# ------------------------------
# INDIVIDUAL EXPERIMENT ANALYSIS
# ------------------------------

for experiment, results in all_results.items():

    # Individual distribution
    plt.figure(figsize=(12, 18))
    for im, metric in enumerate(metrics):
        metric_name = metric[0]
        plt.subplot(4, 2, im + 1)
        plt.hist(results[metric_name], bins=30)
        plt.title(f"{metric[1]} - mean: {np.mean(results[metric_name]):.2f}, std: {np.std(results[metric_name]):.2f}")
    plt.suptitle(f"{experiment}")
    plt.show()

    # Correlation matrix between metrics
    metrics2 = [metrics[i] for i in [0,2,4,1,3,5]]    # reorder metrics for better visualization
    plt.figure(figsize=(10, 8))
    corr = results[[m[0] for m in metrics2]].corr()
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(metrics2)), labels=[m[1] for m in metrics2], rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(metrics2)), labels=[m[1] for m in metrics2])
    # Add correlation coefficients on the matrix
    for i in range(len(metrics2)):
        for j in range(len(metrics2)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
    plt.tight_layout()
    # if SAVE_FIGS:
    #     plt.savefig(f'figures/corr_matrix_{experiment.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Find a proxy score function expressed as - abs(log_prob-t) - k*rep3 maximally correlating with mean llm score
    best_corr_m, best_t_m, best_k_m = -1.0, 0.0, 0.0
    best_corr_h, best_t_h, best_k_h = -1.0, 0.0, 0.0
    t_values = np.linspace(-4.0, 0.0, 41)
    k_values = np.linspace(0.0, 5.0, 51)
    for t in t_values:
        for k in k_values:
            score = - np.abs(results['avg_log_prob'] - t) - k * results['rep3']
            corr_m = np.corrcoef(score, results['mean_llm_score'])[0, 1]
            if corr_m > best_corr_m:
                best_corr_m, best_t_m, best_k_m = corr_m, t, k
            corr_h = np.corrcoef(score, results['harmonic_llm_score'])[0, 1]
            if corr_h > best_corr_h:
                best_corr_h, best_t_h, best_k_h = corr_h, t, k

    print(f"{experiment} Best correlation with mean LLM score: {best_corr_m:.4f} (at t={best_t_m:.2f}, k={best_k_m:.2f})")
    print(f"{experiment} Best correlation with harmonic LLM score: {best_corr_h:.4f} (at t={best_t_h:.2f}, k={best_k_h:.2f})")

# ======================================================================================================================

# -----------------------------
# COMPARISON ACROSS EXPERIMENTS
# -----------------------------
FIG_WIDTH = 18 * np.sqrt(len(folders)/6)
fig, axes = plt.subplots(4, 2, figsize=(FIG_WIDTH, 24))
axes = axes.flatten()

experiments = list(folders.keys())
x = np.arange(len(experiments))
width = 0.6

graph_data = []
for im, metric in enumerate(metrics):
    metric_name = metric[0]
    metric_label = metric[1]

    # Collect means and stds for this metric across all experiments
    means = []
    stds = []
    for experiment, stats in all_stats.items():
        if metric_name in stats:
            means.append(stats[metric_name]['mean'])
            stds.append(stats[metric_name]['std'])
        else:
            means.append(0)
            stds.append(0)

        graph_data.append({
            "metric": metric_name,
            "experiment": experiment,
            "mean": means[-1],
            "std": stds[-1],
        })

    # Create barplot
    ax = axes[im]
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, ecolor=(0.5, 0.5, 0.5, 0.5), color=run_colors)
    ax.set_title(metric_label, fontsize =16)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean - 0.0 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'{mean:.2f}', ha='center', va='bottom', fontsize=16, color='black')

    ax.set_ylim(metric[2][0], metric[2][1])

with open('results/evaluation/evaluation_summary.json', 'w') as f:
    json.dump(graph_data, f, indent=4)

plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('figures/evaluation_summary.svg', dpi=300, bbox_inches='tight')
plt.show()