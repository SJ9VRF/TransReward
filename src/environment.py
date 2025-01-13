import numpy as np

class Environment:
    """
    Simulates an environment in which a robot operates. This includes managing the state of the environment and applying the dynamics of a task to produce new states and rewards based on actions taken by the robot.

    Attributes:
        task (Task): The task being performed in this environment, which includes specific dynamics and a reward function.
        current_state (any): The current state of the environment, which can be of any type defined by the specific task.
    """

    def __init__(self, task):
        """
        Initializes the Environment with a given task.

        Args:
            task (Task): An instance of the Task class which defines the dynamics and reward function for the environment.
        """
        self.task = task
        self.current_state = self.get_initial_state()

    def reset(self):
        """
        Resets the environment to an initial state, which is usually determined by the specifics of the task. This method is typically called at the start of each new simulation or experiment run.

        Returns:
            any: The initial state of the environment.
        """
        self.current_state = self.get_initial_state()
        return self.current_state

    def step(self, action):
        """
        Applies the given action to the environment, calculates the next state using the task's dynamics, and evaluates the reward using the task's reward function.

        Args:
            action (any): The action taken by the robot, the type and structure of which depend on the specifics of the task.

        Returns:
            tuple: A tuple containing the next state and the reward obtained as a result of the action.
        """
        next_state = self.task.dynamics(self.current_state, action)
        reward = self.task.compute_reward(self.current_state, action)
        self.current_state = next_state  # Update the current state to the next state
        return next_state, reward

    def get_initial_state(self):
        """
        Generates an initial state for the environment. This method should be defined based on the requirements of the specific task. It could be a fixed starting point or generated randomly depending on the task's nature.

        Returns:
            any: The initial state of the environment, specific to the task's requirements.
        """
        # Initialize a more complex state for example in a robotic navigation task
        return np.array([0.0, 0.0])  # Placeholder for a 2D coordinate start at the origin

    def __str__(self):
        """
        Returns a string representation of the Environment, generally describing the task it is configured for.

        Returns:
            str: A description of the environment and its current state.
        """
        state_description = f"Current State: {self.current_state}"
        return f"Environment(Task: {self.task}, {state_description})"

