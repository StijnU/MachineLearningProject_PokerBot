import sys

import pyspiel
from open_spiel.python.algorithms import deep_cfr_tf2


def train():
    """Trains the model and save the trained policy"""
    num_iterations = 1000
    num_traversals = 200

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
        game,
        policy_network_layers=(64, 64, 64, 64),
        advantage_network_layers=(64, 64, 64, 64),
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=int(1e4),
        policy_network_train_steps=5000,
        advantage_network_train_steps=500,
        reinitialize_advantage_networks=True,
        infer_device="cpu",
        train_device="cpu")
    deep_cfr_solver.solve()
    deep_cfr_solver.save_policy_network("./fcpa_policy")


if __name__ == "__main__":
    sys.exit(train())
