import unittest
from reward_compensation import RewardCompensation
from robot import Robot
from environment import Environment
from task import Task

class TestRewardCompensation(unittest.TestCase):
    def setUp(self):
        # Define simple task dynamics and reward functions for controlled testing
        self.task = Task(lambda s, a: s + a, lambda s, a: -abs(s - 5))
        self.env = Environment(self.task)
        
        # Create two robots with simple policies
        self.robot1 = Robot("SourceRobot", self.env, lambda s: s + 1)
        self.robot2 = Robot("TargetRobot", self.env, lambda s: s - 1)

        # Perform tasks to generate test data
        self.robot1_data = [(i, i + 1, self.task.compute_reward(i, i + 1)) for i in range(10)]
        self.robot2_data = [(i, i - 1, self.task.compute_reward(i, i - 1)) for i in range(10)]

        # Initialize RewardCompensation with the collected data
        self.reward_compensation = RewardCompensation(self.robot1_data, self.robot2_data)

    def test_calculate_compensation(self):
        # Test if compensation calculation correctly computes the compensation function
        compensation_func = self.reward_compensation.calculate_compensation()
        # Check if the compensation function returns a modified reward as expected
        _, _, test_reward = self.robot1_data[0]
        modified_reward = compensation_func(0, 1, test_reward)
        self.assertNotEqual(test_reward, modified_reward, "Modified reward should differ from the original")

    def test_apply_compensation(self):
        # Apply compensation and check if the target robot's behavior changes as expected
        compensation_func = self.reward_compensation.calculate_compensation()
        self.reward_compensation.apply_compensation(self.robot2, compensation_func)
        
        # Run the task with the compensated robot policy
        compensated_data = self.robot2.perform_task()
        _, _, first_compensated_reward = compensated_data[0]

        # Direct calculation for the expected result
        _, _, original_reward = self.robot2_data[0]
        expected_reward = compensation_func(0, -1, original_reward)
        
        self.assertEqual(first_compensated_reward, expected_reward, "Compensated reward should match expected modified reward")

if __name__ == '__main__':
    unittest.main()

