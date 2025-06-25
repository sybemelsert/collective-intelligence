import itertools
import polars as pl
import datetime
from vi import HeadlessSimulation
from castle_test import SimConfig, Castle, Prey, Predator
from collections import Counter
import csv
import os

def generate_nearby_values(center, max_steps, allowed_values=None, is_int=False):
    """
    Generate up to (2*max_steps + 1) values around center:
    max_steps below, center, max_steps above, clipped to allowed_values if given.
    """
    # Get index of center in allowed_values if available
    if allowed_values:
        if center not in allowed_values:
            # If center not in allowed_values, find closest
            allowed_values = sorted(allowed_values)
            center = min(allowed_values, key=lambda x: abs(x - center))
        center_idx = allowed_values.index(center)
        start_idx = max(0, center_idx - max_steps)
        end_idx = min(len(allowed_values), center_idx + max_steps + 1)
        vals = allowed_values[start_idx:end_idx]
    else:
        # If no allowed_values given, generate range around center
        vals = [center + i for i in range(-max_steps, max_steps + 1)]
        if is_int:
            vals = [int(round(v)) for v in vals]
        vals = sorted(set(vals))
    return vals

def run_stability_tests():
    # Your 3 best known configs:
    best_configs = [
        {"prey_prob": 0.002, "predator_death": 0.004, "initial_prey": 60, "initial_predator": 20},
        {"prey_prob": 0.0015, "predator_death": 0.004, "initial_prey": 40, "initial_predator": 10},
        {"prey_prob": 0.001, "predator_death": 0.003, "initial_prey": 40, "initial_predator": 10},
    ]

    max_steps = 2  # number of steps below and above center to include

    # Allowed discrete values for parameters (taken from your previous runs)
    prey_prob_allowed = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
    predator_death_allowed = [0.002, 0.003, 0.004, 0.005]
    initial_prey_allowed = [20, 30, 40, 50, 60, 70, 80]
    initial_predator_allowed = [5, 10, 15, 20, 25, 30]

    num_runs_per_config = 3
    results = []

    # Create output folder if not exists
    output_folder = "Assignment_2/test_results"
    os.makedirs(output_folder, exist_ok=True)

    # Build configs with neighborhoods around base configs
    configs_to_test = []
    for base in best_configs:
        prey_probs = generate_nearby_values(base["prey_prob"], max_steps, prey_prob_allowed)
        predator_deaths = generate_nearby_values(base["predator_death"], max_steps, predator_death_allowed)
        initial_preys = generate_nearby_values(base["initial_prey"], max_steps, initial_prey_allowed, is_int=True)
        initial_predators = generate_nearby_values(base["initial_predator"], max_steps, initial_predator_allowed, is_int=True)

        for combo in itertools.product(prey_probs, predator_deaths, initial_preys, initial_predators):
            config_dict = {
                "prey_prob": combo[0],
                "predator_death": combo[1],
                "initial_prey": combo[2],
                "initial_predator": combo[3],
            }
            if config_dict not in configs_to_test:
                configs_to_test.append(config_dict)

    print(f"Total configs to test: {len(configs_to_test)}")

    for config_i, config in enumerate(configs_to_test, 1):
        prey_prob = config["prey_prob"]
        predator_death = config["predator_death"]
        initial_prey = config["initial_prey"]
        initial_predator = config["initial_predator"]

        predator_counts = []
        zero_count_runs = 0

        print(f"\nTesting config {config_i}/{len(configs_to_test)}: prey_prob={prey_prob}, predator_death={predator_death}, prey={initial_prey}, predator={initial_predator}")

        for run_i in range(num_runs_per_config):
            sim_config = SimConfig(
                duration=60 * 60 * 1,  # 1 minute simulation
                prey_reproduction_prob=prey_prob,
                predator_death_prob=predator_death,
            )

            df = (
                HeadlessSimulation(config=sim_config)
                .spawn_agent(Castle, images=["Assignment_2/images/fort.png"])
                .batch_spawn_agents(initial_prey, Prey, images=["Assignment_2/images/prey_small.png"])
                .batch_spawn_agents(initial_predator, Predator, images=["Assignment_2/images/predator_small.png"])
                .run()
                .snapshots
            )

            max_frame = df.select(pl.col("frame").max()).item()
            last_frame_df = df.filter(pl.col("frame") == max_frame)
            final_predator_count = last_frame_df.filter(pl.col("kind") == "Predator").height
            predator_counts.append(final_predator_count)

            survived = final_predator_count > 0
            if not survived:
                zero_count_runs += 1

            print(f"  Run {run_i + 1}: Final predator count = {final_predator_count}")

            # Early stop if more than half runs are zeros so far
            if zero_count_runs > num_runs_per_config / 2:
                print("  More than half runs ended with zero predators — skipping further runs for this config.")
                break

        avg_pred_count = sum(predator_counts) / len(predator_counts)
        survival_rate = (len(predator_counts) - zero_count_runs) / len(predator_counts)

        results.append({
            "prey_prob": prey_prob,
            "predator_death": predator_death,
            "initial_prey": initial_prey,
            "initial_predator": initial_predator,
            "avg_final_predator_count": avg_pred_count,
            "predator_survival_rate": survival_rate,
            "runs_completed": len(predator_counts),
        })

    # Save summary CSV
    summary_csv_path = os.path.join(output_folder, "stability_test_summary.csv")
    with open(summary_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "prey_prob", "predator_death", "initial_prey", "initial_predator",
            "avg_final_predator_count", "predator_survival_rate", "runs_completed"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSaved summary results to {summary_csv_path}")

    # Filter stable configs (>= 80% survival)
    stable_configs = [r for r in results if r["predator_survival_rate"] >= 0.8]

    if stable_configs:
        best_config = max(stable_configs, key=lambda x: x["avg_final_predator_count"])
        print("\nBest stable config with high predator survival:")
        print(best_config)

        prey_prob_vals = [c["prey_prob"] for c in stable_configs]
        predator_death_vals = [c["predator_death"] for c in stable_configs]
        initial_prey_vals = [c["initial_prey"] for c in stable_configs]
        initial_predator_vals = [c["initial_predator"] for c in stable_configs]

        def most_common(lst):
            return Counter(lst).most_common(1)[0][0]

        print("\nBest parameter values from stable configs:")
        print(f"prey_prob: {most_common(prey_prob_vals)}")
        print(f"predator_death: {most_common(predator_death_vals)}")
        print(f"initial_prey: {most_common(initial_prey_vals)}")
        print(f"initial_predator: {most_common(initial_predator_vals)}")

    else:
        print("\nNo stable configs found with predator survival >= 80%")

if __name__ == "__main__":
    run_stability_tests()
