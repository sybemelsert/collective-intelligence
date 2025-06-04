from dataclasses import dataclass
from vi import Agent, Config, Simulation
from pygame.math import Vector2
from vi.config import deserialize
import random


@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5
    delta_time: float = 1 # reduced for smoother motion
    mass: int = 20
    max_velocity: float = 2.0

    def weights(self) -> tuple[float, float, float]:
        return (
            self.alignment_weight,
            self.cohesion_weight,
            self.separation_weight,
        )


class FlockingAgent(Agent):
    # By overriding `change_position`, the default behaviour is overwritten.
    # Without making changes, the agents won't move.
    def change_position(self):
        neighbors = self.in_proximity_accuracy()
        neighbors = [agent for agent, _ in neighbors]

        # If no neighbors, give a small nudge to avoid freezing
        if not neighbors:
            if self.move.length() < 0.01:
                angle = random.uniform(0, 360)
                self.move = Vector2(1, 0).rotate(angle) * 0.5
            self.pos += self.move * self.config.delta_time
            self._wrap_position()
            return

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

        # Get speed after force update
        speed = self.move.length()
        min_speed = 0.5  # <-- enforce this as the minimum movement threshold
        max_speed = self.config.max_velocity

        # Reassign velocity with proper normalization and limits
        if speed < min_speed:
            self.move = self.move.normalize() * min_speed
        elif speed > max_speed:
            self.move = self.move.normalize() * max_speed
        # Update position
        self.pos += self.move * delta_time
        self._wrap_position()
    

    def _wrap_position(self):
        # Keep agents inside the window (wraparound)
        width, height =1000, 1000 
        self.pos.x %= width
        self.pos.y %= height


# Run the simulation
(
    Simulation(
        FlockingConfig(
            image_rotation=True,
            movement_speed=2.0,
            radius=80,
            fps_limit=0
        )
    )
    .batch_spawn_agents(100, FlockingAgent, images=["Assignment_0/images/triangle.png"])
    .run()
)
