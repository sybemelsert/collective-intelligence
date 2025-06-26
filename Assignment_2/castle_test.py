from dataclasses import dataclass
from vi import Agent, Config, HeadlessSimulation
import random
import polars as pl
import matplotlib.pyplot as plt
import datetime
import math
from pygame.math import Vector2

# Global prey count
TOTAL_PREY = 0

@dataclass
class SimConfig(Config):
    radius: float = 10
    speed: float = 1.0
    prey_reproduction_prob: float = 0.0015
    predator_death_prob: float = 0.005
    predator_reproduction_chance: float = 1
    castle_capacity: int = 10
    max_castle_stay: int = 60 * 3   # 3 minutes if 60 FPS
    castle_radius: float = 50
    repel_strength: float = 1.5
    detection_radius: float = 100
    eating_radius: float = 15


class Castle(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preys_in_castle = {}
        self.config.radius = self.config.castle_radius

    def update(self):
        self.save_data('kind', 'Castle')

    def allow_entry(self, prey):
        return len(self.preys_in_castle) < self.config.castle_capacity

    def enter(self, prey):
        self.preys_in_castle[prey] = 0
        angle = random.uniform(0, 2 * math.pi)
        min_distance = prey.config.radius
        max_distance = self.config.castle_radius - (prey.config.radius / 0.7)
        distance = random.uniform(min_distance, max_distance)
        prey.pos = self.pos + Vector2(math.cos(angle), math.sin(angle)) * distance
        prey.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()

    def leave(self, prey):
        self.preys_in_castle.pop(prey, None)

    def tick(self):
        to_remove = []
        for prey, ticks in self.preys_in_castle.items():
            if not prey.alive or ticks >= self.config.max_castle_stay:
                to_remove.append(prey)
            else:
                self.preys_in_castle[prey] += 1
        for prey in to_remove:
            self.preys_in_castle.pop(prey, None)


class Prey(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_castle = False
        self.castle_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed
        self.current_castle = None

    def kill(self):
        global TOTAL_PREY
        if self.alive:
            TOTAL_PREY = max(0, TOTAL_PREY - 1)
        super().kill()

    def update(self):
        global TOTAL_PREY
        if self.in_castle:
            self.save_data('kind', 'Prey')
            self.castle_timer += 1
            if self.castle_timer >= self.config.max_castle_stay:
                self.leave_castle()
            return
        else:
            self.save_data('kind', 'Prey')

        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                castle = agent
                if (self.pos - castle.pos).length() <= castle.config.castle_radius and castle.allow_entry(self):
                    self.enter_castle(castle)
                    return
                repel = (self.pos - castle.pos)
                if repel.length() > 0:
                    repel = repel.normalize()
                self.move += repel * self.config.repel_strength

        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        # Reproduce only if TOTAL_PREY < 400
        if TOTAL_PREY < 400 and random.random() < self.config.prey_reproduction_prob:
            self.reproduce()
            TOTAL_PREY += 1

    def enter_castle(self, castle):
        self.in_castle = True
        self.castle_timer = 0
        self.current_castle = castle
        castle.enter(self)
        self.move = Vector2(0, 0)

    def leave_castle(self):
        if self.current_castle:
            self.current_castle.leave(self)
            repel_vector = (self.pos - self.current_castle.pos)
            if repel_vector.length() > 0:
                repel_vector = repel_vector.normalize()
            else:
                repel_vector = Vector2(1, 0)
            self.pos = self.current_castle.pos + repel_vector * (self.config.castle_radius + self.config.radius)
            self.move = repel_vector * self.config.speed * 1.5
        self.in_castle = False
        self.castle_timer = 0
        self.current_castle = None


class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def update(self):
        global TOTAL_PREY
        self.save_data('kind', 'Predator')

        # Castle avoidance
        for agent, dist in self.in_proximity_accuracy().filter_kind(Castle):
            if dist < self.config.detection_radius:
                castle = agent
                repel = (self.pos - castle.pos)
                if repel.length() > 0:
                    repel = repel.normalize()
                self.move += repel * self.config.repel_strength * 2.0

        self.move = self.move.normalize() * self.config.speed
        self.pos += self.move

        # Hunting
        for prey, dist in self.in_proximity_accuracy().filter_kind(Prey):
            if not prey.in_castle and dist < self.config.eating_radius:
                prey.kill()
                TOTAL_PREY = max(0, TOTAL_PREY - 1)
                self.has_eaten = True
                if random.random() < self.config.predator_reproduction_chance:
                    self.reproduce()
                break

        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()

        self.has_eaten = False


# Run multiple simulation runs, reset TOTAL_PREY at start of each
num_runs = 30
for i in range(num_runs):
    print(f"\n=== Running simulation {i + 1} of {num_runs} ===")
    TOTAL_PREY = 0  # Reset prey count before each run

    result_df = (
        HeadlessSimulation(config=SimConfig(duration=60 * 60 * 1))
        .spawn_agent(Castle, images=["images/fort.png"])
        .batch_spawn_agents(50, Prey, images=["images/prey_small.png"])
        .batch_spawn_agents(25, Predator, images=["images/predator_small.png"])
        .run()
        .snapshots
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{i+1}_{timestamp}"
    csv_path = f"Assignment_2/test_results/base_case/baseCastle{run_id}.csv"
    plot1_path = f"Assignment_2/test_results/week2/baseCastle{run_id}.png"

    result_df.write_csv(csv_path)
    print(f"✅ Saved snapshot data to {csv_path}")

    df_grouped = (
        result_df
        .group_by(["frame", "kind"])
        .agg(pl.count("id").alias("count"))
        .pivot(index="frame", values="count", on="kind")
        .fill_null(0)
        .sort("frame")
    )

    df_grouped = df_grouped.with_columns(
        (pl.col("frame") / 60).alias("time_seconds")
    )

    # Consistent colors for agents
    color_map = {
        "Prey": "#1f77b4",
        "Predator": "#d62728",
        "Castle": "#2ca02c",
    }

    plt.figure(figsize=(10, 6))
    for kind in df_grouped.columns[1:-1]:  # exclude 'frame' and 'time_seconds'
        plt.plot(
            df_grouped["time_seconds"],
            df_grouped[kind],
            label=kind,
            color=color_map.get(kind, None)
        )

    plt.title("Population Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    print(f"✅ Saved population plot to {plot1_path}")
