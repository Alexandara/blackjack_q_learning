"""
Blackjack Reinforcement Learning
"""
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
        self.q_table = {}
        for p_sum in range(4,32):
            for dealer_card in range(1,12):
                for usable_ace in [0,1]:
                    self.q_table[(p_sum, dealer_card, usable_ace)] = {}
                    for action in [0,1]:
                        if (p_sum == 21) and (action == 0):
                            self.q_table[(p_sum, dealer_card, usable_ace)][action] = 1
                        else:
                            self.q_table[(p_sum, dealer_card, usable_ace)][action] = 0
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
        err = self.reward + self.discount_factor * (future_optimal - self.q_table[self.observation][action])
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
        x_axis = []
        for i in range(len(self.training_error)):
            x_axis.append(i)
        plt.plot(x_axis, self.training_error)
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.title('Training Error')
        plt.show()

if __name__ == '__main__':
    bql = BlackjackQLearning()
    bql.train()
    avg_reward = bql.test(100000)
    print("Avg Reward:", avg_reward)
    bql.plot_reward()
    bql.plot_training_error()
