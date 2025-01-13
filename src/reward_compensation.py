class RewardCompensation:
    """
    Manages the calculation and application of reward compensation between different robots or tasks.
    
    Attributes:
        source_data (list): Collected data from the source robot or task, containing tuples of (state, action, reward).
        target_data (list): Collected data from the target robot or task, similar to source_data.
    """

    def __init__(self, source_data, target_data):
        """
        Initializes the RewardCompensation object with data from both source and target robots or tasks.

        Args:
            source_data (list): Data from the source task, each item is a tuple (state, action, reward).
            target_data (list): Data from the target task, each item is a tuple (state, action, reward).
        """
        self.source_data = source_data
        self.target_data = target_data

    def calculate_compensation(self):
        """
        Calculates a compensation function based on the differences in reward distributions between the source and target data.

        Returns:
            function: A compensation function that can be used to adjust actions or rewards.
        """
        # Calculate average rewards for source and target
        source_avg_reward = sum([data[2] for data in self.source_data]) / len(self.source_data)
        target_avg_reward = sum([data[2] for data in self.target_data]) / len(self.target_data)

        # Determine the compensation factor as the difference in average rewards
        compensation_factor = source_avg_reward - target_avg_reward

        # Create a compensation function that adjusts the reward by this factor
        def compensation_func(state, action, reward):
            return reward + compensation_factor

        return compensation_func

    def apply_compensation(self, robot, compensation_func):
        """
        Applies the calculated reward compensation to adjust the robot's policy or reward mechanism.

        Args:
            robot (Robot): The robot to which the compensation will be applied.
            compensation_func (function): The compensation function to adjust the robot's rewards.
        """
        original_policy = robot.policy

        def compensated_policy(state):
            action = original_policy(state)
            _, reward = robot.environment.step(action)
            compensated_reward = compensation_func(state, action, reward)
            return action, compensated_reward

        # Set the new policy with compensation on the robot
        robot.policy = compensated_policy

    def __str__(self):
        """
        Returns a string representation of the RewardCompensation, showing basic info about data lengths.

        Returns:
            str: A brief summary of the RewardCompensation instance.
        """
        return f"RewardCompensation(Source Data Length: {len(self.source_data)}, Target Data Length: {len(self.target_data)})"

