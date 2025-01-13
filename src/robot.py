import numpy as np

class Robot:
    """
    Represents a robotic agent capable of performing tasks in a given environment.
    
    Attributes:
        name (str): The name of the robot.
        environment (Environment): The environment in which the robot operates.
        policy (callable): A function that takes a state and returns an action.
        data (list): A list to store tuples of (state, action, reward) from task performances.
    """

    def __init__(self, name, environment, initial_policy):
        """
        Initializes a Robot with a given name, environment, and an initial policy.

        Args:
            name (str): The name of the robot.
            environment (Environment): The environment object in which the robot will operate.
            initial_policy (callable): A function that defines the robot's decision-making strategy.
        """
        self.name = name
        self.environment = environment
        self.policy = initial_policy
        self.data = []

    def perform_task(self):
        """
        Simulates the robot performing its task in the environment by following its policy.
        Collects data on states, actions, and rewards.

        Returns:
            list: A list of tuples containing (state, action, reward) for each step performed.
        """
        self.data.clear()  # Clear previous task data
        state = self.environment.reset()  # Reset the environment and get the initial state
        for _ in range(100):  # Perform 100 steps in the environment
            action = self.policy(state)  # Determine action based on the current state and policy
            next_state, reward = self.environment.step(action)  # Take the action in the environment
            self.data.append((state, action, reward))  # Store the state, action, and reward
            state = next_state  # Update the current state to the next state
        return self.data

    def adjust_policy(self, compensation_func):
        """
        Adjusts the robot's policy based on a compensation function provided externally,
        typically for reward compensation adjustments.

        Args:
            compensation_func (callable): A function that modifies the action based on the state.
        """
        original_policy = self.policy  # Keep the original policy

        # Define a new policy that incorporates the compensation
        def compensated_policy(state):
            original_action = original_policy(state)  # Get the original action
            compensated_action = compensation_func(state, original_action)  # Apply compensation
            return compensated_action

        self.policy = compensated_policy  # Set the new compensated policy

    def __str__(self):
        """
        Returns a string representation of the robot.

        Returns:
            str: A description of the robot including its name.
        """
        return f"Robot(name={self.name})"

