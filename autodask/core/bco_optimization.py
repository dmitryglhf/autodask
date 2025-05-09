import time
import numpy as np

import warnings

from utils.log import get_logger

warnings.filterwarnings('ignore')


class BeeColonyOptimizer:
    def __init__(self,
                 employed_bees=20,
                 onlooker_bees=10,
                 scout_bees=5,
                 abandonment_limit=10,
                 exploration_rate=0.3):
        self.employed_bees = employed_bees
        self.onlooker_bees = onlooker_bees
        self.scout_bees = scout_bees
        self.abandonment_limit = abandonment_limit
        self.exploration_rate = exploration_rate
        self.solutions = []
        self.solution_trials = {}

        self.log = get_logger(self.__class__.__name__)

    def optimize(self, model_class, param_space, X_train, y_train, X_val, y_val,
                 metric_func, maximize=True, rounds=30, time_limit=None):
        """
        Implement bee colony optimization for hyperparameter tuning
        """
        start_time = time.time()
        best_params = None
        best_score = float('-inf') if maximize else float('inf')

        # Initialize solutions with random parameter sets
        for _ in range(self.employed_bees):
            params = self._generate_random_params(param_space)
            score = self._evaluate_params(model_class, params, X_train, y_train, X_val, y_val, metric_func)
            self.solutions.append((params, score))
            self.solution_trials[str(params)] = 0

            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params

        for round_idx in range(rounds):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # Employed bees phase: explore neighborhood of existing solutions
            for i in range(len(self.solutions)):
                params, score = self.solutions[i]
                new_params = self._explore_neighborhood(params, param_space)
                new_score = self._evaluate_params(model_class, new_params, X_train, y_train, X_val, y_val, metric_func)

                if (maximize and new_score > score) or (not maximize and new_score < score):
                    self.solutions[i] = (new_params, new_score)
                    self.solution_trials[str(params)] = 0
                    if (maximize and new_score > best_score) or (not maximize and new_score < best_score):
                        best_score = new_score
                        best_params = new_params
                else:
                    self.solution_trials[str(params)] += 1

            # Onlooker bees phase: focus on promising solutions
            fitness_values = [s[1] for s in self.solutions]
            if maximize:
                probs = [max(0, f - min(fitness_values) + 1e-10) for f in fitness_values]
            else:
                probs = [max(0, max(fitness_values) - f + 1e-10) for f in fitness_values]

            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            for _ in range(self.onlooker_bees):
                selected_idx = np.random.choice(len(self.solutions), p=probs)
                params, score = self.solutions[selected_idx]
                new_params = self._explore_neighborhood(params, param_space)
                new_score = self._evaluate_params(
                    model_class, new_params, X_train, y_train, X_val, y_val, metric_func
                )

                if (maximize and new_score > score) or (not maximize and new_score < score):
                    self.solutions[selected_idx] = (new_params, new_score)
                    self.solution_trials[str(params)] = 0
                    if (maximize and new_score > best_score) or (not maximize and new_score < best_score):
                        best_score = new_score
                        best_params = new_params
                else:
                    self.solution_trials[str(params)] += 1

            # Scout bees phase: abandon solutions that haven't improved and explore new areas
            for i in range(len(self.solutions)):
                params, _ = self.solutions[i]
                if self.solution_trials[str(params)] >= self.abandonment_limit:
                    new_params = self._generate_random_params(param_space)
                    new_score = self._evaluate_params(model_class, new_params, X_train, y_train, X_val, y_val,
                                                      metric_func)
                    self.solutions[i] = (new_params, new_score)
                    self.solution_trials[str(new_params)] = 0

                    if (maximize and new_score > best_score) or (not maximize and new_score < best_score):
                        best_score = new_score
                        best_params = new_params

        return best_params, best_score

    def _generate_random_params(self, param_space):
        """Generate random parameters from the parameter space"""
        params = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
            else:
                raise ValueError(f"Unsupported parameter range format for {param_name}")
        return params

    def _explore_neighborhood(self, params, param_space):
        """Explore the neighborhood of a solution"""
        new_params = params.copy()
        param_to_change = np.random.choice(list(params.keys()))

        if np.random.random() < self.exploration_rate:
            # Explore randomly
            new_params[param_to_change] = self._generate_random_params(
                {param_to_change: param_space[param_to_change]})[
                param_to_change]
        else:
            # Make a small adjustment
            param_range = param_space[param_to_change]
            if isinstance(param_range, list):
                new_params[param_to_change] = np.random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                current_value = params[param_to_change]
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    delta = np.random.choice([-1, 1])
                    new_value = current_value + delta
                    new_params[param_to_change] = max(param_range[0], min(param_range[1], new_value))
                else:
                    delta = (param_range[1] - param_range[0]) * 0.1 * np.random.choice([-1, 1])
                    new_value = current_value + delta
                    new_params[param_to_change] = max(param_range[0], min(param_range[1], new_value))

        return new_params

    def _evaluate_params(self, model_class, params, X_train, y_train, X_val, y_val, metric_func):
        """Evaluate a parameter set by training and validating a model"""
        try:
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = metric_func(y_val, y_pred)
            return score
        except Exception as e:
            self.log.info(f"Error evaluating params {params}: {e}")
            return float('-inf')  # Return a very bad score for invalid parameters
