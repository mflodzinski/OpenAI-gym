import numpy as np
import gymnasium as gym


class CartPole:
    def __init__(
        self,
        num_bins,
        num_runs,
        epsilon,
        show_every,
        learn_rate,
        discount,
        start_epsilon_decay,
        end_epsilon_decay,
        q_table=None,
    ):
        self.num_bins = num_bins
        self.num_runs = num_runs
        self.epsilon = epsilon
        self.show_every = show_every
        self.learn_rate = learn_rate
        self.discount = discount
        self.start_epsilon_decay = start_epsilon_decay
        self.end_epsilon_decay = end_epsilon_decay
        self.epsilon_decay_value = epsilon / (end_epsilon_decay - start_epsilon_decay)
        self.env = self.init_env()
        self.bins = self.create_bins(num_bins)
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table = self.create_q_table(num_bins)

    def init_env(self):
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset()
        return env

    def create_bins(self, num_bins):
        bins = [
            np.linspace(-4.8, 4.8, num_bins),
            np.linspace(-2, 2, num_bins),
            np.linspace(-0.418, 0.418, num_bins),
            np.linspace(-0.873, 0.873, num_bins),
        ]
        return bins

    def create_q_table(self, num_bins):
        obs_space_size = len(self.env.observation_space.low)
        num_act = self.env.action_space.n
        q_table = np.random.uniform(
            low=0, high=1, size=([num_bins] * obs_space_size + [num_act])
        )
        return q_table

    def get_discrete_state(self, state):
        state_index = []
        for sub_state, bin in zip(state, self.bins):
            state_index.append(np.digitize(sub_state, bin) - 1)
        return tuple(state_index)

    def change_render_mode(self, to_human):
        if to_human:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1", render_mode="rgb_array")

    def q_learning(self, does_render):
        for i in range(1, self.num_runs + 1):
            if does_render and i % self.show_every == 0:
                self.change_render_mode(to_human=True)

            state, _ = self.env.reset()
            discrete_state = self.get_discrete_state(state)
            done = False
            cnt = 0

            while not done:
                cnt += 1

                if np.random.random() > self.epsilon:
                    action = np.argmax(self.q_table[discrete_state])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, *_ = self.env.step(action)
                new_discrete_state = self.get_discrete_state(new_state)
                max_future_q = np.max(self.q_table[new_discrete_state])
                curr_q = self.q_table[discrete_state + (action,)]

                if done and cnt < 200:
                    reward -= 375

                new_q = curr_q + self.learn_rate * (
                    reward + self.discount * max_future_q - curr_q
                )
                self.q_table[discrete_state + (action,)] = new_q
                discrete_state = new_discrete_state

            if does_render and i % self.show_every == 0:
                self.change_render_mode(to_human=False)
            if i % (self.show_every / 10) == 0:
                print(f"-> Epoch no.{i}")
            if self.end_epsilon_decay >= i >= self.start_epsilon_decay:
                self.epsilon -= self.epsilon_decay_value

        np.save("./data/q_table.npy", self.q_table)

    def run_simulation(self, runs):
        for _ in range(runs):
            self.change_render_mode(to_human=True)
            state, _ = self.env.reset()
            discrete_state = self.get_discrete_state(state)
            done = False
            cnt = 0

            while not done:
                action = np.argmax(self.q_table[discrete_state])
                new_state, reward, done, *_ = self.env.step(action)
                discrete_state = self.get_discrete_state(new_state)
                cnt += 1

            print(cnt)
