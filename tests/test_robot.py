import unittest
from robot import Robot
from environment import Environment
from task import Task

class TestRobot(unittest.TestCase):
    def setUp(self):
        # Setup a simple task and environment for testing
        self.task = Task(lambda s, a: s + a, lambda s, a: -abs(s - a))
        self.env = Environment(self.task)
        self.robot = Robot("TestRobot", self.env, lambda s: s + 1)  # Robot with a simple policy of always incrementing state by 1

    def test_perform_task(self):
        # Test that performing the task results in expected changes in state and reward collection
        data = self.robot.perform_task()
        self.assertEqual(len(data), 100, "Should perform 100 iterations")
        # Check the first transition
        first_state, first_action, first_reward = data[0]
        self.assertEqual(first_state + first_action, data[1][0], "Next state should be the sum of state and action")
        self.assertEqual(-abs(first_state - first_action), first_reward, "Reward should be negative absolute difference")

    def test_adjust_policy(self):
        # Test adjusting the robot's policy
        def compensation_func(state, action):
            return action + 10  # Simple compensation that always adds 10 to the action

        self.robot.adjust_policy(compensation_func)
        data = self.robot.perform_task()
        first_state, first_action, _ = data[0]
        expected_action = compensation_func(first_state, first_state + 1)
        self.assertEqual(first_action, expected_action, "Action should be adjusted by the compensation function")

    def test_initial_state(self):
        # Test the initial state setting of the environment
        initial_state = self.robot.environment.reset()
        self.assertEqual(initial_state, 0, "Initial state should be 0 as defined in Environment.get_initial_state")

if __name__ == '__main__':
    unittest.main()

