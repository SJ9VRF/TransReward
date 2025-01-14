import unittest
from task import Task

class TestTask(unittest.TestCase):
    def setUp(self):
        # Setup simple dynamics and a reward function for the task
        self.dynamics = lambda s, a: s + a
        self.reward_function = lambda s, a: 100 - abs(s - a)  # Rewards decrease with distance from target

        # Initialize the task
        self.task = Task(self.dynamics, self.reward_function)

    def test_dynamics(self):
        # Test dynamics to ensure they correctly modify the state based on an action
        state = 0
        action = 5
        expected_next_state = self.dynamics(state, action)
        self.assertEqual(self.task.dynamics(state, action), expected_next_state, "Dynamics function did not produce expected next state")

    def test_compute_reward(self):
        # Test reward computation to ensure it calculates the correct reward based on state and action
        state = 0
        action = 5
        expected_reward = self.reward_function(state, action)
        self.assertEqual(self.task.compute_reward(state, action), expected_reward, "Reward function did not compute the expected reward")

    def test_simulate_step(self):
        # Test simulate_step to ensure it returns the correct state and reward tuple
        state = 0
        action = 5
        next_state = self.dynamics(state, action)
        reward = self.reward_function(state, action)
        
        simulated_next_state, simulated_reward = self.task.simulate_step(state, action)
        
        self.assertEqual(simulated_next_state, next_state, "Simulate step did not return the correct next state")
        self.assertEqual(simulated_reward, reward, "Simulate step did not return the correct reward")

if __name__ == '__main__':
    unittest.main()

