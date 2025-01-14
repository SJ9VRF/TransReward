import unittest
from environment import Environment
from task import Task

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Setup simple task dynamics and reward function for the environment
        self.dynamics = lambda s, a: s + 2 * a  # Dynamics that doubles the action effect
        self.reward_function = lambda s, a: 100 - abs(s - a)  # Reward decreases as the difference between s and a increases

        # Initialize the task and environment
        self.task = Task(self.dynamics, self.reward_function)
        self.environment = Environment(self.task)

    def test_reset(self):
        # Test reset method to ensure it correctly initializes the environment
        initial_state = self.environment.reset()
        self.assertIsNotNone(initial_state, "Initial state should not be None")
        self.assertEqual(initial_state, 0, "Initial state should be correctly set by the environment")

    def test_step(self):
        # Test step method to ensure it processes actions and updates state correctly
        initial_state = self.environment.reset()
        action = 5
        expected_next_state = self.dynamics(initial_state, action)
        expected_reward = self.reward_function(initial_state, action)
        
        next_state, reward = self.environment.step(action)
        
        self.assertEqual(next_state, expected_next_state, "Next state should be correctly updated by the dynamics function")
        self.assertEqual(reward, expected_reward, "Reward should be correctly calculated by the reward function")

    def test_get_initial_state(self):
        # Test the get_initial_state method to ensure it returns a correct and consistent initial state
        initial_state = self.environment.get_initial_state()
        self.assertEqual(initial_state, 0, "get_initial_state should return the correct initial state")

if __name__ == '__main__':
    unittest.main()

