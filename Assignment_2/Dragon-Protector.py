from dataclasses import dataclass
from vi import Agent, Config, Simulation
import random

# --- Simulation configuration ---
@dataclass
class SimConfig(Config):
    radius: float = 10
    speed: float = 1.0
    prey_reproduction_prob: float = 0.001
    predator_death_prob: float = 0.005
    predator_reproduction_chance: float = 1
    duration: int = 60 * 60 * 0.5  # 30-minute simulated time

# --- Prey agent ---
class Prey(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "Prey")  # log position & kind
        if random.random() < self.config.prey_reproduction_prob:
            self.reproduce()

# --- Predator agent ---
class Predator(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_eaten = False

    def update(self):
        self.pos += self.move
        self.save_data('kind', "Predator")  # log position & kind

        prey = (
            self.in_proximity_accuracy()
            .without_distance()
            .filter_kind(Prey)
            .first()
        )
        if prey is not None:
            prey.kill()
            self.has_eaten = True
            if random.random() < self.config.predator_reproduction_chance:
                self.reproduce()

        if not self.has_eaten and random.random() < self.config.predator_death_prob:
            self.kill()
        self.has_eaten = False

# --- Protector Dragon agent ---
class ProtectorDragon(Agent):
    def update(self):
        self.pos += self.move
        self.save_data('kind', "ProtectorDragon")  # log position & kind

        nearby_prey = self.in_proximity_accuracy().filter_kind(Prey)
        for prey_tuple in nearby_prey:
            prey_agent = prey_tuple[0]
            nearby_predators = prey_agent.in_proximity_accuracy().filter_kind(Predator)
            for predator in nearby_predators:
                predator.kill()

# --- Run simulation ---
result_df = (
    Simulation(config=SimConfig())
    .batch_spawn_agents(60, Prey, images=["images/prey_small.png"])
    .batch_spawn_agents(20, Predator, images=["images/predator_small.png"])
    .batch_spawn_agents(1, ProtectorDragon, images=["images/shield_small.png"])
    .run()
    .snapshots
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Classic Lotka-Volterra parameters
alpha = 0.1   # Prey birth rate
beta = 0.02   # Predation rate
delta = 0.01  # Predator reproduction rate
gamma = 0.1   # Predator death rate

# Dragon phase parameters (no predation, no predator reproduction)
gamma_dragon = 0.15  # Faster predator death due to dragon
alpha_dragon = 0.2   # Faster prey growth (no predators)

# Time grid
t = np.linspace(0, 200, 2000)

# Define ODEs
def lotka_volterra(state, t):
    x, y = state
    if t < 100:
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
    else:
        dxdt = alpha_dragon * x  # Exponential prey growth
        dydt = -gamma_dragon * y  # Predator decline
    return [dxdt, dydt]

# Initial populations
x0 = 40  # Initial prey
y0 = 9   # Initial predators
initial_state = [x0, y0]

# Solve ODE
solution = odeint(lotka_volterra, initial_state, t)
prey, predators = solution.T

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, prey, label="Prey Population ", linewidth=2)
plt.plot(t, predators, label="Predator Population ", linewidth=2)
plt.axvline(x=100, color='red', linestyle='--', label=' Dragon Appears (t=100)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Dynamics with Dragon Intervention')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
