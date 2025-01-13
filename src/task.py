class Task:
    """
    Represents a specific task with defined dynamics and a reward function.
    
    Attributes:
        dynamics (callable): A function that takes a state and an action and returns the next state.
        reward_function (callable): A function that takes a state and an action and returns a reward.
    """

    def __init__(self, dynamics, reward_function):
        """
        Initializes a Task with specified dynamics and reward function.

        Args:
            dynamics (callable): The function defining how the state changes in response to an action.
            reward_function (callable): The function that calculates the reward based on state and action.
        """
        self.dynamics = dynamics
        self.reward_function = reward_function

    def compute_reward(self, state, action):
        """
        Computes the reward for a given state and action using the task's reward function.

        Args:
            state (any): The current state in the task environment.
            action (any): The action taken in the current state.

        Returns:
            float: The reward resulting from the given state and action.
        """
        return self.reward_function(state, action)

    def simulate_step(self, state, action):
        """
        Simulates a single step in the task by applying the dynamics to the current state and action.

        Args:
            state (any): The current state in the task environment.
            action (any): The action taken in the current state.

        Returns:
            tuple: A tuple containing the next state and the reward for the step.
        """
        next_state = self.dynamics(state, action)
        reward = self.compute_reward(state, action)
        return next_state, reward

    def __str__(self):
        """
        Returns a string representation of the Task, typically showing the type of dynamics and reward structure.

        Returns:
            str: A description of the task.
        """
        return f"Task(Dynamics={self.dynamics.__name__}, Reward Function={self.reward_function.__name__})"

