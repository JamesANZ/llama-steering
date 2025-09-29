import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

folder = "results/sweep_1D/20251018-051852_sweep_1D" # Baseline L15

with open(os.path.join(folder, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
    description = config["description"]
print(description)

with open(os.path.join(folder, "results.json"), "r") as f:
    all_df = pd.read_json(f)
    all_df['surprise'] = -all_df['avg_log_prob']
    all_df['eiffel'] = all_df['answer'].str.lower().str.contains('eiffel').astype(int)
    all_df['arithmetic_mean'] = (all_df['llm_score_concept'] + all_df['llm_score_instruction'] + all_df['llm_score_fluency']) / 3
    all_df['harmonic_mean'] = 3 / (1/all_df['llm_score_concept'] + 1/all_df['llm_score_instruction'] + 1/all_df['llm_score_fluency'])


quantities = [
    ("llm_score_concept", "LLM concept score", (0.0, 2)),
    ("eiffel", "Explicit concept inclusion", (0, 1)),
    ("llm_score_instruction", "LLM instruction score", (0.0, 2)),
    ("surprise", "Surprise in reference model", (0.0, 3.0)),
    ("llm_score_fluency", "LLM fluency score", (0.0, 2)),
    ("rep3", "3-gram repetition", (0.0, 1)),
    ("arithmetic_mean", "Arithmetic mean of LLM scores", (0.0, 2)),
    ("harmonic_mean", "Harmonic mean of LLM scores", (0.0, 2)),
]

# List of unique (feature, index) pairs
features = all_df[["layer", "feature_index"]].drop_duplicates().values

for feature in features:

    layer_idx, feature_idx = feature
    feature_name = f"L{layer_idx}F{feature_idx}"
    df = all_df[(all_df["layer"] == layer_idx) & (all_df["feature_index"] == feature_idx)]
    print("Analyzing feature:", feature_name)

    # Compute statistics for each quantity
    stats = {}
    for col, _, _ in quantities:
        stats[col] = {
            "median": df.groupby("steering_intensity")[col].median().reset_index(),
            "p10": df.groupby("steering_intensity")[col].quantile(0.1).reset_index(),
            "p90": df.groupby("steering_intensity")[col].quantile(0.9).reset_index(),
            "mean": df.groupby("steering_intensity")[col].mean().reset_index(),
            "std": df.groupby("steering_intensity")[col].std().reset_index(),
        }

    # Save stats to CSV by flattening the nested structure
    rows = []
    for quantity in stats.keys():
        for stat_type in stats[quantity].keys():
            stat_df = stats[quantity][stat_type]
            for _, row in stat_df.iterrows():
                rows.append({
                    "steering_intensity": row["steering_intensity"],
                    "quantity": quantity,
                    "stat_type": stat_type,
                    "value": row[quantity]
                })
    stats_csv_df = pd.DataFrame(rows)
    stats_csv_path = os.path.join(folder, f"stats_{feature_name}.csv")
    stats_csv_df.to_csv(stats_csv_path, index=False)
    print(f"Saved stats to {stats_csv_path}")

    # Plot statistics in a single figure as a function of steering intensity
    plt.figure(figsize=(12, 20))
    for i, (col, ylabel, ylim) in enumerate(quantities, 1):
        plt.subplot(4, 2, i)
        plt.fill_between(
            stats[col]["std"].steering_intensity,
            stats[col]["mean"][col] + stats[col]["std"][col],
            stats[col]["mean"][col] - stats[col]["std"][col],
            color="lightblue",
            label="± 1 std dev",
            )
        plt.plot(
            stats[col]["mean"].steering_intensity,
            stats[col]["mean"][col],
            "-ob",
            label="Mean",
        )
        plt.legend()
        plt.xlim(0, max(df['steering_intensity']))
        plt.ylim(*ylim)
        plt.xlabel("Steering intensity")
        plt.title(f"{ylabel}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"figures/{description}_{feature_name}_sweep_1D_all_metrics.png")
    plt.show()