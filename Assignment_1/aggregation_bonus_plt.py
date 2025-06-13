from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
import random, math, datetime
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class AggregationConfig(Config):
    speed: float = 0.5
    radius: float = 10.0
    aggregation_zone_radius: float = 120.0
    Tjoin: int = 30
    Tleave: int = 30

class AggregationAgent(Agent):
    WANDERING, JOIN, STILL, LEAVE = range(4)
    zone = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = self.WANDERING
        self.state_timer = 0

    def initialise_agent(self):
        self.state = self.WANDERING
        self.state_timer = 0
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.speed

    def change_position(self):
        neighbors = self.in_proximity_accuracy()
        n = sum(1 for _, dist in neighbors if dist < self.config.aggregation_zone_radius)
        in_zone = n > 0

        a, b = 1.70188, 3.88785
        PJoin = 0.03 + 0.48 * (1 - math.exp(-a * n)) if in_zone else 0
        PLeave = math.exp(-b * n) if in_zone else 1

        if self.state == self.WANDERING:
            if random.random() < 0.02:
                angle = random.uniform(-45, 45)
                self.move = self.move.rotate(angle)
                if self.move.length() == 0:
                    self.move = Vector2(1, 0)
                self.move = self.move.normalize() * self.config.speed
            if in_zone and random.random() < PJoin:
                self.state, self.state_timer = self.JOIN, 0
            self.pos += self.move

        elif self.state == self.JOIN:
            self.state_timer += 1
            if self.state_timer > self.config.Tjoin:
                self.state, self.state_timer = self.STILL, 0
            self.pos += self.move.normalize() * self.config.speed * 0.2

        elif self.state == self.STILL:
            self.move = Vector2(0, 0)
            self.state_timer += 1
            if self.state_timer > self.config.Tleave and random.random() < PLeave:
                self.state, self.state_timer = self.LEAVE, 0

        elif self.state == self.LEAVE:
            if self.move.length() < 0.01:
                angle = random.uniform(0, 360)
                self.move = Vector2(1, 0).rotate(angle).normalize() * self.config.speed
            self.state_timer += 1
            if self.state_timer > 10:
                self.state, self.state_timer = self.WANDERING, 0
            self.pos += self.move

        self._wrap_position()
        self.save_data("state", self.state)


    def _wrap_position(self):
        self.pos.x %= 1000
        self.pos.y %= 1000

# Run the simulation and access snapshots
sim = (
    Simulation(
        AggregationConfig(
            image_rotation=True,
            speed=1,
            radius=10,
            fps_limit=0,
            duration=1000 * 60,
        )
    )
    .batch_spawn_agents(100, AggregationAgent, images=["Assignment_1/images/triangle.png"])
    .run()
)

# Get agent states per frame from snapshots
df = (
    sim.snapshots
    .group_by(["frame", "state"])
    .agg(pl.count("id").alias("agents"))
)

# Optional: smoothing
window_size = 300
df = df.sort(["state", "frame"])
df = df.with_columns(
    pl.col("agents")
    .rolling_mean(window_size, min_periods=1)
    .over("state")
    .alias("agents_smoothed")
)

# Map state indices to names for plotting
state_names = {0: "WANDERING", 1: "JOIN", 2: "STILL", 3: "LEAVE"}
df = df.with_columns(
    pl.col("state").map_elements(lambda s: state_names.get(s, str(s))).alias("state_name")
)

# Save data
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"Assignment_1/results_stage2/bonus{timestamp}.png"
data_filename = f"Assignment_1/results_stage2/State_data_{timestamp}.csv"

df.write_csv(data_filename)

# Plot
sns.set(style="darkgrid")
plot = sns.relplot(data=df.to_pandas(), x="frame", y="agents_smoothed", hue="state_name", kind="line")
plot.set_titles("Agent States Over Time")
plot.savefig(plot_filename, dpi=300)

print(f"Saved plot to {plot_filename} and data to {data_filename}")
