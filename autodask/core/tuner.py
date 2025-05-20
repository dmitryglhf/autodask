import time
import numpy as np
import warnings

from autodask.utils.log import get_logger
from autodask.utils.cross_val import evaluate_model

warnings.filterwarnings('ignore')


class BeeColonyOptimizer:
    """Implementation of the Bee Colony Optimization (BCO) algorithm for hyperparameter tuning."""

    def __init__(self,
                 task: str,
                 employed_bees=3,
                 onlooker_bees=3,
                 exploration_rate=0.3,
                 cv_folds=2):
        self.employed_bees = employed_bees
        self.onlooker_bees = onlooker_bees
        self.exploration_rate = exploration_rate

        self.solutions = None
        self.solution_trials = None

        self.task = task
        self.cv_folds = cv_folds

        self.log = get_logger(self.__class__.__name__)

    def optimize(self, model_class, param_space, X_train, y_train,
                 metric_func, maximize=True, rounds=1, time_limit=None):
        """Run the bee colony optimization algorithm."""
        start_time = time.time()
        best_params = None
        best_score = float('-inf') if maximize else float('inf')

        # Colony initialization and initial solution update
        self.log.info("Colony initialization...")
        self.initialize_colony(model_class, param_space, X_train, y_train, metric_func)
        best_params, best_score = self.update_best_solution(best_params, best_score, maximize)

        # Optimization loop
        self.log.info("Starting optimization loop")
        for round_idx in range(rounds):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # Employed bees phase
            self.employed_bees_phase(model_class, param_space, X_train, y_train, metric_func, maximize)
            best_params, best_score = self.update_best_solution(best_params, best_score, maximize)

            # Onlooker bees phase
            self.onlooker_bees_phase(model_class, param_space, X_train, y_train, metric_func, maximize)
            best_params, best_score = self.update_best_solution(best_params, best_score, maximize)

        self.log.info(f"Obtained parameters: {best_params}")
        self.log.info(f"Best score: {round(best_score, 3)}")
        return best_params, best_score

    def initialize_colony(self, model_class, param_space, X_train, y_train, metric_func):
        """Initialize bee colony with random solutions.

        Args:
            model_class: The machine learning model class to optimize
            param_space (dict): Parameter search space
            X_train: Training features
            y_train: Training targets
            metric_func (callable): Scoring function

        Returns:
            None: Updates solutions and solution_trials attributes
        """
        self.solutions = []
        self.solution_trials = {}

        for _ in range(self.employed_bees):
            params = self._generate_random_params(param_space)
            score = evaluate_model(model_class, params, X_train, y_train, metric_func, self.task, self.cv_folds)

            # Create a stable key for the params
            params_key = self._get_params_key(params)

            self.solutions.append((params, score))
            self.solution_trials[params_key] = 0

    def update_best_solution(self, best_params, best_score, maximize):
        """Updates best solution.

        Args:
            best_params: Current best parameters
            best_score: Current best score
            maximize (bool): Whether to maximize or minimize the score

        Returns:
            tuple: Updated (best_params, best_score)
        """
        for params, score in self.solutions:
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params

        return best_params, best_score

    def employed_bees_phase(self, model_class, param_space, X_train, y_train, metric_func, maximize):
        """Employed bees phase: explore neighborhood of existing solutions.

        Args:
            model_class: The machine learning model class to optimize
            param_space (dict): Parameter search space
            X_train: Training features
            y_train: Training targets
            metric_func (callable): Scoring function
            maximize (bool): Whether to maximize or minimize the score

        Returns:
            None: Updates solutions and solution_trials attributes
        """
        for i in range(len(self.solutions)):
            params, score = self.solutions[i]
            params_key = self._get_params_key(params)

            new_params = self._explore_neighborhood(params, param_space)
            new_params_key = self._get_params_key(new_params)
            new_score = evaluate_model(
                model_class, new_params, X_train, y_train, metric_func, self.task, self.cv_folds
            )

            if (maximize and new_score > score) or (not maximize and new_score < score):
                self.solutions[i] = (new_params, new_score)
                self.solution_trials[new_params_key] = 0
            else:
                # Initialize if key doesn't exist
                if params_key not in self.solution_trials:
                    self.solution_trials[params_key] = 0
                self.solution_trials[params_key] += 1

    def _calculate_selection_probabilities(self, maximize):
        """Calculate probabilities for onlooker bees based on solution quality.

        Args:
            maximize (bool): Whether to maximize or minimize the score

        Returns:
            list: Probability distribution for solution selection
        """
        fitness_values = [s[1] for s in self.solutions]

        if maximize:
            probs = [max(0, f - min(fitness_values) + 1e-10) for f in fitness_values]
        else:
            probs = [max(0, max(fitness_values) - f + 1e-10) for f in fitness_values]

        total = sum(probs)
        if total > 0:
            return [p / total for p in probs]
        return [1 / len(probs) for _ in probs]  # Равномерное распределение, если все значения одинаковы

    def onlooker_bees_phase(self, model_class, param_space, X_train, y_train, metric_func, maximize):
        """Onlooker bees phase: focus on promising solutions.

        Args:
            model_class: The machine learning model class to optimize
            param_space (dict): Parameter search space
            X_train: Training features
            y_train: Training targets
            metric_func (callable): Scoring function
            maximize (bool): Whether to maximize or minimize the score

        Returns:
            None: Updates solutions and solution_trials attributes
        """
        probs = self._calculate_selection_probabilities(maximize)

        for _ in range(self.onlooker_bees):
            if len(probs) > 0:  # Only proceed if we have valid probabilities
                selected_idx = np.random.choice(len(self.solutions), p=probs)
                params, score = self.solutions[selected_idx]
                params_key = self._get_params_key(params)

                new_params = self._explore_neighborhood(params, param_space)
                new_params_key = self._get_params_key(new_params)
                new_score = evaluate_model(
                    model_class, new_params, X_train, y_train, metric_func, self.task, self.cv_folds
                )

                if (maximize and new_score > score) or (not maximize and new_score < score):
                    self.solutions[selected_idx] = (new_params, new_score)
                    self.solution_trials[new_params_key] = 0
                else:
                    # Initialize if key doesn't exist
                    if params_key not in self.solution_trials:
                        self.solution_trials[params_key] = 0
                    self.solution_trials[params_key] += 1

    def _get_params_key(self, params):
        """Create a stable hashable key for parameter dictionaries."""
        # Sort items to ensure consistent string representation
        items = sorted(params.items())
        # Create a tuple of tuples that can be used as a dictionary key
        return str(items)

    def _generate_random_params(self, param_space):
        """Generate random parameters from the parameter space."""
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
        """Generate a new solution by exploring neighborhood of current solution."""
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
