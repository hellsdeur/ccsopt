from typing import List, Dict, Tuple
import numpy as np

from ccsopt.predictor import CCSPredictor

class PSOCCS:
    def __init__(
            self, num_particles: int, num_iterations: int, inertia_weight: int, cognitive_coeff: int, social_coeff: int,
            feature_names: List[str], component_names: List[str], bounds: Dict[str, Tuple[float, float]],
            penalty_lambda: float, target_sum: float, predictor: CCSPredictor,  seed: int = 42):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.feature_names = feature_names
        self.component_names = component_names
        self.bounds = bounds
        self.penalty_lambda = penalty_lambda
        self.target_sum = target_sum
        self.predictor = predictor
        self.rng = np.random.RandomState(seed)

        self.dimension = len(self.feature_names)
        self.lower_bounds = np.array([bounds[feature][0] for feature in self.feature_names])
        self.upper_bounds = np.array([bounds[feature][1] for feature in self.feature_names])

        self.vmax = 0.2 * (self.upper_bounds - self.lower_bounds)

    def initialize(self):
        self.positions = (
                self.lower_bounds +
                (self.upper_bounds - self.lower_bounds) *
                self.rng.rand(self.num_particles, self.dimension)
        )

        self.velocities = np.zeros_like(self.positions)

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.array([self.objective_function(pos) for pos in self.positions])

        self.global_best_index = np.argmin(self.personal_best_values)
        self.global_best_position = self.personal_best_positions[self.global_best_index].copy()
        self.global_best_value = self.personal_best_values[self.global_best_index]

        self.history = []

    def run(self):
        self.initialize()

        for iteration in range(self.num_iterations):
            for i in range(self.num_particles):
                r1, r2 = self.rng.rand(), self.rng.rand()

                cognitive_factor = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_factor = self.social_coeff * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = (
                        self.inertia_weight * self.velocities[i] +
                        cognitive_factor +
                        social_factor
                )

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bounds, self.upper_bounds)

                current_value = self.objective_function(self.positions[i])

                if current_value > self.personal_best_values[i]:
                    self.personal_best_positions[i] = self.positions[i].copy()
                    self.personal_best_values[i] = current_value

                    if current_value > self.global_best_value:
                        self.global_best_position = self.personal_best_positions[i].copy()
                        self.global_best_value = current_value

            self.history.append(self.global_best_value)

        return self.global_best_position, self.global_best_value, self.history

    def objective_function(self, position):
        components = position[:len(self.component_names)]
        mix_sum = np.sum(components)
        penalty = self.penalty_lambda * (mix_sum - self.target_sum) ** 2
        age_index = self.feature_names.index("age")
        position[age_index] = int(position[age_index])
        strength = self.predictor.predict(*position)
        return strength - penalty
