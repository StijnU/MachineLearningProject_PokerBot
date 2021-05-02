import sys

import pyspiel
from open_spiel.python.algorithms import deep_cfr_tf2


def train():
    """Trains the model and save the trained policy"""
    num_iterations = 100
    num_traversals = 10

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
        game,
        policy_network_layers=(1024, 256),
        advantage_network_layers=(1024, 256),
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=1024,
        memory_capacity=int(1e5),
        policy_network_train_steps=int(1e3),
        advantage_network_train_steps=200,
        reinitialize_advantage_networks=False)
    deep_cfr_solver.solve()
    deep_cfr_solver.save_policy_network("./fcpa_policy")


if __name__ == "__main__":
    sys.exit(train())
