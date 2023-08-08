import json
import numpy as np
from cartpole import CartPole


if __name__ == "__main__":
    with open("./data/cartpole_parameters.json", "r") as json_file:
        params = json.load(json_file)
    try:
        q_table = np.load("./data/q_table.npy")
    except FileNotFoundError:
        q_table = None

    cart_pole = CartPole(**params, q_table=q_table)
    cart_pole.q_learning(does_render=False)
    cart_pole.run_simulation(runs=5)
