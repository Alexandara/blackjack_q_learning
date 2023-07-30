"""
Blackjack Reinforcement Learning
"""
from __future__ import annotations

from collections import defaultdict

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class BlackjackQLearning:
    """
    BlackjackQLearning creates a reinforcement learning model that trains to
    play blackjack
    """
    def __init__(self, render_mode=None, epochs=100000, learning_rate=.0001, discount_factor=.95, epsilon=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.start_e = epsilon
        # OpenAI Gymnasium
        self.env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=render_mode)
        self.observation, self.info = self.env.reset()
        # Q-Table Initialization
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.training_error = []

    def train(self):
        """
        Method to train the model
        """
        for _ in range(self.epochs):
            self.run_game()
            self.reduce_epsilon()

    def run_game(self, update=True):
        """
        Runs a single game of blackjack.
        :return: observation and reward at end of game
        """
        terminated = False
        truncated = False
        self.observation, self.info = self.env.reset()
        while not truncated and not terminated:
            action = self.select_action()
            temp_observation, self.reward, terminated, truncated, self.info = self.env.step(action)
            if update:
                self.update_model(temp_observation, action, terminated)
            self.observation = temp_observation

    def select_action(self):
        """
        Method that uses the model to select a new action
        :return: either 0 or 1 wherein 1 is hit and 0 is stick
        """
        # Randomly select a state
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.choice([0,1])
        # Or use what we've learned
        return int(max(self.q_table[self.observation][0],self.q_table[self.observation][1]))

    def reduce_epsilon(self):
        self.epsilon = max(.1, self.epsilon - (self.start_e / (self.epochs / 2)))

    def update_model(self, temp_observation, action, terminated):
        """
        Method that updates the Q-Table for the model
        :param observation: current world state
        :param reward: current reward
        """
        future_optimal = (not terminated) * max(self.q_table[temp_observation][0], self.q_table[temp_observation][1])
        self.q_table[self.observation][action] = self.q_table[self.observation][action] + \
                                         self.learning_rate * (self.reward + self.discount_factor * future_optimal - self.q_table[self.observation][action])
        err = self.reward + self.discount_factor * future_optimal
        self.training_error.append(err)

    def test(self, test_rounds=100):
        """
        Function that tests the model over a certain amount of games, returning
        the average reward score.
        :return: average reward score over the games
        """
        sum_rewards = 0.0
        for _ in range(test_rounds):
            self.run_game(update=False)
            sum_rewards = sum_rewards + self.reward
        return sum_rewards / test_rounds

    def plot_training_error(self):
        _, axis = plt.subplots(figsize=(6, 5))
        axis.set_title("Training Error Over Iterations")
        error = (
                np.convolve(np.array(self.training_error), np.ones(500), mode="same")
                / 500
        )
        axis.plot(range(len(error)), error)

        plt.tight_layout()
        plt.savefig("TrainingError.png")

def tabulate_results(file, epochs=100000, learning_rate=.0001, discount_factor=.95, epsilon=1):
    bql = BlackjackQLearning(epochs=epochs, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)
    bql.train()
    averages = []
    for i in range(10):
        averages.append(bql.test(100))
    ans = round(sum(averages) / len(averages), 2)
    file.write(str(epochs) + "," + str(learning_rate) + "," + \
               str(discount_factor) + "," + str(epsilon) + \
               "," + str(ans) + "\n")


if __name__ == '__main__':
    results_file = open("results.csv", "a")
    results_file.write("Epochs,Learning Rate,Discount Factor,Epsilon,Average Win Rate\n")
    print("Epoch Testing")
    epochs = 10
    while epochs <= 1000000:
        tabulate_results(results_file, epochs=epochs)
        epochs = epochs * 10
    print("Learning Rate Testing")
    learning_rate = .1
    while learning_rate >= .00000001:
        tabulate_results(results_file, learning_rate=learning_rate)
        learning_rate = learning_rate * .1
    print("Discount Rate Testing")
    discount_factor = 1
    while discount_factor >= .3:
        tabulate_results(results_file, discount_factor=discount_factor)
        discount_factor = discount_factor - .05
    print("Epsilon Testing")
    epsilon = 1
    while epsilon >= .5:
        tabulate_results(results_file, epsilon=epsilon)
        epsilon = epsilon - .05
    results_file.close()
    bql = BlackjackQLearning(epochs=100000, learning_rate=.0000001, discount_factor=.35, epsilon=.6)
    bql.train()
    print(bql.test())
    bql.plot_training_error()
