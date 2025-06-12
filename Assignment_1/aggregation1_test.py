from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import math
import pygame
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# ------------------------------
# CONFIGURATION
# ------------------------------
@dataclass
class AggregationConfig(Config):
    speed: float = 0.5
    radius: float = 10.0
    aggregation_zone_radius: float = 120.0
    Tjoin: int = 30
    Tleave: int = 30

# ------------------------------
# AGGREGATION ZONE
# ------------------------------
class AggregationZone:
    def __init__(self, pos: Vector2, radius: float):
        self.pos = pos
        self.radius = radius

# ------------------------------
# AGENT DEFINITION
# ------------------------------
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

    def update(self):
        # Draw aggregation zone (called every frame)
        screen = pygame.display.get_surface()
        if screen and AggregationAgent.zone:
            pos = AggregationAgent.zone.pos
            radius = AggregationAgent.zone.radius
            pygame.draw.circle(screen, (255, 255, 0), (int(pos.x), int(pos.y)), int(radius), width=2)

    def change_position(self):
        zone = self.zone
        neighbors = self.in_proximity_accuracy()

        if zone:
            in_zone = (self.pos - zone.pos).length() < zone.radius
            n = sum(1 for agent, _ in neighbors if (agent.pos - zone.pos).length() < zone.radius)
        else:
            n = sum(1 for _, dist in neighbors if dist < self.config.aggregation_zone_radius)
            in_zone = n > 0
        self.in_zone = in_zone  
        self.state_name = ["WANDERING", "JOIN", "STILL", "LEAVE"][self.state]  # <--- For easier analysis
        # Probability formulas
        a, b = 1.70188, 3.88785
        PJoin = 0.03 + 0.48 * (1 - math.exp(-a * n)) if in_zone else 0
        PLeave = math.exp(-b * n) if in_zone else 1
        if in_zone:
            self.change_image(1) # assign the alternate image index when inside the zone
        else:
            self.change_image(0) # default image index when outside

        # State transitions
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

    def _wrap_position(self):
        self.pos.x %= 1000
        self.pos.y %= 1000

# ------------------------------
# RUN SIMULATION  
# ------------------------------
AggregationAgent.zone = AggregationZone(Vector2(500, 500), 120)
for run in range(1, 2): # Change range for more runs
    df = (
        HeadlessSimulation(
            AggregationConfig(
                image_rotation=True,
                speed=10,
                radius=10,
                fps_limit=0,
                duration=1000 * 60, # Length of simulation
            )
        )
        .batch_spawn_agents(
            100,
            AggregationAgent,
            images=["Assignment_1/images/triangle.png", "Assignment_1/images/triangle_zone.png"]
        )
        .run()
        .snapshots
        .group_by(["frame", "image_index"])
        .agg(pl.count("id").alias("agents"))
    )
    window_size = 10 # Adjust window size for smoothing
    df = df.sort(["image_index", "frame"])
    df = df.with_columns(
        pl.col("agents")
        .rolling_mean(window_size, min_periods=1)
        .over("image_index")
        .alias("agents_smoothed")
    )

    # Generate unique filenames using timestamp and run number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"Assignment_1/results_stage1/aggregation_plot_smoothed_{timestamp}_run{run}.png"
    data_filename = f"Assignment_1/results_stage1/aggregation_data_{timestamp}_run{run}.csv"

    print(df)
    df.write_csv(data_filename)  # Save table as CSV

    plot = sns.relplot(x=df['frame'], y=df['agents_smoothed'], hue=df['image_index'], kind="line")
    plot.savefig(plot_filename, dpi=300)
    print(f"Saved plot to {plot_filename} and data to {data_filename}")