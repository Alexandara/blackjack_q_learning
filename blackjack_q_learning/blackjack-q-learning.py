"""
Blackjack Reinforcement Learning
"""
import gymnasium as gym
import numpy as np

class BlackjackQLearning:
    """
    BlackjackQLearning creates a reinforcement learning model that trains to
    play blackjack
    """
    def __init__(self, render_mode=None, epochs=100, learning_rate=.1, discount_factor=.5):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # OpenAI Gymnasium
        self.env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=render_mode)
        self.observation, self.info = self.env.reset()
        # Q-Table Initialization
        self.q_table = {}
        for p_sum in range(4,30):
            for dealer_card in range(2,11):
                for usable_ace in range(0,1):
                    self.q_table[(p_sum, dealer_card, usable_ace)] = {}
                    for action in range(0,1):
                        if (p_sum == 21) and (action == 0):
                            self.q_table[(p_sum, dealer_card, usable_ace)][action] = 1
                        else:
                            self.q_table[(p_sum, dealer_card, usable_ace)][action] = 0

        self.state = (0, 0, False)  # initial state
        self.actions = [1, 0]  # 1: HIT  0: STAND

    def train(self):
        """
        Method to train the model
        """
        for _ in range(self.epochs):
            temp_observation, reward = self.run_game()
            self.update_model(temp_observation, reward)

    def run_game(self):
        """
        Runs a single game of blackjack.
        :return: observation and reward at end of game
        """
        terminated = False
        truncated = False
        reward = 0
        temp_observation = self.observation
        while not truncated and not terminated:
            action = self.select_action()
            temp_observation, reward, terminated, truncated, self.info = self.env.step(action)
            if terminated or truncated:
                self.observation, self.info = self.env.reset()
        return temp_observation, reward

    def select_action(self):
        """
        Method that uses the model to select a new action
        :return: either 0 or 1 wherein 1 is hit and 0 is stick
        """
        # TODO: Replace with model action selection
        # if the current value <= 11, always hit b/c there is no harm in hitting another card
        current_val = self.state[0]
        if current_val <= 11:
            return 1

        # If the card sum is > 11 then discount_factor% of the time the player takes a random action
        # otherwise the player takes a greedy approach which is the action that gains the
        # most reward based on the current estimates of Q-value.
        if np.random.uniform(0, 1) <= self.discount_factor:
            action = np.random.choice(self.actions)
        else:
            # greedy approach
            v = -999
            action = 0
            for i in self.q_table[self.state]:
                if self.q_table[self.state][i] > v:
                    action = i
                    v = self.q_table[self.state][i]
        return action

    def update_model(self,observation, reward):
        """
        Method that updates the Q-Table for the model
        :param observation: current world state
        :param reward: current reward
        """
        # TODO: Use observation and the reward to update the model
        print("update")

    def test(self, test_rounds=100):
        """
        Function that tests the model over a certain amount of games, returning
        the average reward score.
        :param test_rounds: rounds of the game to play
        :return: average reward score over the games
        """
        sum_rewards = 0
        for _ in range(test_rounds):
            _, reward = self.run_game()
            sum_rewards = sum_rewards + reward
        return sum_rewards / test_rounds


if __name__ == '__main__':
    bql = BlackjackQLearning()
    print(bql.test())
