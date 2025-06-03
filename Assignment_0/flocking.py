from dataclasses import dataclass

from vi import Agent, Config, Simulation
from pygame.math import Vector2

@dataclass
class FlockingConfig(Config):
    # TODO: Modify the weights and observe the change in behaviour.
    alignment_weight: float = 1
    cohesion_weight: float = 1
    separation_weight: float = 1


class FlockingAgent(Agent):
    def change_position(self):
        neighbors = self.in_proximity_accuracy()
        neighbors = [agent for agent, _ in neighbors]

        if not neighbors:
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
        f_c = x_avg - x_boid
        cohesion = f_c - v_boid

        # Weights & constants
        α = self.config.alignment_weight
        β = self.config.separation_weight
        γ = self.config.cohesion_weight
        mass = getattr(self.config, "mass", 20)
        delta_time = getattr(self.config, "delta_time", 3)
        max_speed = getattr(self.config, "max_velocity", 2.0)

        # Total force
        f_total = (α * alignment + β * separation + γ * cohesion) / mass

        # Velocity update
        self.move += f_total

        # Apply min(Move.length, MaxVelocity) * Move.normalize()
        speed = min(self.move.length(), max_speed)
        self.move = self.move.normalize() * speed if self.move.length() != 0 else Vector2()

        # Position update
        self.pos += self.move * delta_time


        


(
    Simulation(
        # TODO: Modify `movement_speed` and `radius` and observe the change in behaviour.
        FlockingConfig(image_rotation=True, movement_speed=1, radius=50)
    )
    .batch_spawn_agents(100, FlockingAgent, images=["Assignment_0/images/triangle.png"])
    .run()
)
