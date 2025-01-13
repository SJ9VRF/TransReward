import numpy as np

class Experiment:
    """
    Facilitates the setup, execution, and analysis of complex robotic learning experiments. This includes single-robot experiments, multi-robot transfer learning, and multi-robot co-training scenarios with reward compensation.

    Attributes:
        robots (list): A list of robots participating in the experiment.
    """

    def __init__(self, robots):
        """
        Initialize the Experiment with a list of robots.
        
        Args:
            robots (list of Robot): Robots to be included in the experiment.
        """
        self.robots = robots

    def run_single_robot(self, robot, iterations=100):
        """
        Executes tasks for a single robot.
        
        Args:
            robot (Robot): The robot to run the experiment on.
            iterations (int): Number of iterations the robot should perform its task.

        Returns:
            list: Collected data from the robot's task performance.
        """
        return robot.perform_task(iterations)

    def run_multi_robot(self):
        """
        Executes tasks for multiple robots, collecting and returning data without applying any learning transfer or compensation.

        Returns:
            dict: A dictionary with robot names as keys and collected data as values.
        """
        results = {}
        for robot in self.robots:
            data = robot.perform_task()
            results[robot.name] = data
            print(f"Data collected for {robot.name}: {data[:5]}")
        return results

    def run_co_training(self, learning_rate=0.1):
        """
        Executes a co-training scenario where robots learn simultaneously, potentially influencing each other's learning process through shared insights.

        Args:
            learning_rate (float): Learning rate for adjusting policies based on shared insights.

        Returns:
            dict: Co-training results with adjustments made from shared data insights.
        """
        results = {}
        # Simulate a learning step where each robot adjusts its policy based on others' data
        shared_data = np.mean([np.mean(robot.perform_task(), axis=0) for robot in self.robots], axis=0)
        for robot in self.robots:
            robot.adjust_policy(lambda state, action: action + learning_rate * (shared_data - state))
            results[robot.name] = robot.perform_task()
            print(f"Post-co-training data for {robot.name}: {results[robot.name][:5]}")
        return results

    def apply_reward_compensation(self, source_robot, target_robot):
        """
        Applies a reward compensation based on the performance data of the source robot to improve the target robot's learning efficiency.

        Args:
            source_robot (Robot): The robot whose data serves as the basis for compensation.
            target_robot (Robot): The robot whose policy will be adjusted.
        """
        source_avg_reward = np.mean([r[2] for r in source_robot.data])
        target_avg_reward = np.mean([r[2] for r in target_robot.data])
        compensation_factor = source_avg_reward - target_avg_reward

        def compensation_function(state, action):
            return action + compensation_factor

        target_robot.adjust_policy(compensation_function)

def main():
    from robot import Robot
    from environment import Environment
    from task import Task

    # Setting up the environment and tasks
    task = Task(lambda s, a: s + a, lambda s, a: -np.sqrt(abs(s - a)))
    env = Environment(task)

    # Robots with different initial policies
    robot1 = Robot("Robot1", env, lambda s: s + 1)
    robot2 = Robot("Robot2", env, lambda s: s * 2)

    experiment = Experiment([robot1, robot2])

    # Run different types of experiments
    print("Running single robot experiment:")
    experiment.run_single_robot(robot1)

    print("\nRunning multi-robot experiment:")
    experiment.run_multi_robot()

    print("\nRunning co-training experiment:")
    experiment.run_co_training()

    # Applying reward compensation from robot1 to robot2
    print("\nApplying reward compensation:")
    experiment.apply_reward_compensation(robot1, robot2)
    experiment.run_single_robot(robot2)  # Re-run experiment for robot2 after compensation

if __name__ == "__main__":
    main()

