from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
from vi.config import deserialize
import random
import pygame

@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5
    delta_time: float = 1
    mass: int = 20
    max_velocity: float = 2.0

    def weights(self) -> tuple[float, float, float]:
        return (
            self.alignment_weight,
            self.cohesion_weight,
            self.separation_weight,
        )


# Obstacle definition (not visible by itself)
class Obstacle:
    def __init__(self, pos: Vector2, image_path: str, radius: float):
        self.pos = pos
        self.radius = radius
        self.image_path = image_path


# Main agent with flocking + obstacle avoidance
class FlockingAgent(Agent):
    obstacle = None  # Shared by all agents

    def change_position(self):
        neighbors = self.in_proximity_accuracy()
        neighbors = [agent for agent, _ in neighbors]

        if not neighbors:
            if self.move.length() < 0.01:
                angle = random.uniform(0, 360)
                self.move = Vector2(1, 0).rotate(angle) * 0.5
            self.pos += self.move * self.config.delta_time
            self._wrap_position()
            return

        # Obstacle avoidance
        # Obstacle collision handling (hard boundary)
        obstacle = FlockingAgent.obstacle
        if obstacle:
            offset = self.pos - obstacle.pos
            distance = offset.length()
            avoid_distance = obstacle.radius + self.config.radius

            if distance < avoid_distance:
                if distance == 0:
                    offset = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                    distance = offset.length()

                # Normalize the bounce direction
                normal = offset.normalize()

                # Reflect velocity off the obstacle
                self.move = self.move.reflect(normal)

                # Move agent just outside the obstacle radius
                self.pos = obstacle.pos + normal * avoid_distance


        v_boid = self.move
        x_boid = self.pos

        # Alignment
        v_avg = sum((agent.move for agent in neighbors), Vector2()) / len(neighbors)
        alignment = v_avg - v_boid

        # Separation
        separation = sum((x_boid - agent.pos for agent in neighbors), Vector2()) / len(neighbors)

        # Cohesion
        x_avg = sum((agent.pos for agent in neighbors), Vector2()) / len(neighbors)
        cohesion = (x_avg - x_boid) - v_boid

        # Force calculation
        α = self.config.alignment_weight
        β = self.config.separation_weight
        γ = self.config.cohesion_weight
        mass = self.config.mass
        delta_time = self.config.delta_time
        max_speed = self.config.max_velocity

        f_total = (α * alignment + β * separation + γ * cohesion) / mass
        self.move += f_total

        # Clamp velocity
        speed = self.move.length()
        min_speed = 0.5
        if speed < min_speed:
            self.move = self.move.normalize() * min_speed
        elif speed > max_speed:
            self.move = self.move.normalize() * max_speed

        self.pos += self.move * delta_time
        self._wrap_position()

    def _wrap_position(self):
        width, height = 1000, 1000
        self.pos.x %= width
        self.pos.y %= height


# Obstacle agent (visible but doesn't move)
class ObstacleAgent(Agent):
    def initialise_agent(self):
         self.pos = FlockingAgent.obstacle.pos - Vector2(80, 71)

    def change_position(self):
        pass


# Create the obstacle and assign it to agents
obstacle = Obstacle(Vector2(500, 500), "Assignment_0/images/triangle@200px.png", radius=100)
FlockingAgent.obstacle = obstacle


# Start the simulation
sim = Simulation(
    FlockingConfig(
        image_rotation=True,
        movement_speed=2.0,
        radius=80,
        fps_limit=0,
    )
)

# Spawn visible obstacle agent (1 only)
# Add the visual obstacle
sim.batch_spawn_agents(
    1,
    ObstacleAgent,
    images=["Assignment_0/images/triangle@200px.png"]
)

# Add the flocking agents
sim.batch_spawn_agents(
    100,
    FlockingAgent,
    images=["Assignment_0/images/triangle.png"]  # or triangle@50px.png
).run()

