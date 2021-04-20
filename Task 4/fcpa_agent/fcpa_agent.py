#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots, rcfr
import tensorflow.compat.v1 as tf

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        self.policy = None
        fcpa_game_string = pyspiel.hunl_game_string("fcpa")
        game = pyspiel.load_game(fcpa_game_string)
        self.model = rcfr.DeepRcfrModel(
            game,
            num_hidden_layers=1,
            num_hidden_units=13,
            num_hidden_factors=8,
            use_skip_connections=True)
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.state = None
        self.previousAction = [None, None]

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        self.state = state

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        self.state = state
        self.previousAction[player_id] = action

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        if len(state.legal_actions()) == 0:
            return pyspiel.INVALID_ACTION
        features = rcfr.sequence_features(state, num_distinct_actions=state.num_distinct_actions())
        action = self.model(features)
        self.previousAction[self.player_id] = action
        return action

    def train(self):
        """Trains the model and save the trained policy"""
        buffer_size = -1
        fcpa_game_string = pyspiel.hunl_game_string("fcpa")
        game = pyspiel.load_game(fcpa_game_string)
        models = [self.model, rcfr.DeepRcfrModel(
            game,
            num_hidden_layers=1,
            num_hidden_units=13,
            num_hidden_factors=8,
            use_skip_connections=True)]
        if buffer_size > 0:
            solver = rcfr.ReservoirRcfrSolver(
                game,
                models,
                buffer_size,
                truncate_negative=False)
        else:
            solver = rcfr.RcfrSolver(
                game,
                models,
                truncate_negative=False,
                bootstrap=False)

        def _train_fn(model, data):
            """Train `model` on `data`."""
            data = data.shuffle(100 * 10)
            data = data.batch(100)
            data = data.repeat(200)

            optimizer = tf.keras.optimizers.Adam(lr=0.01, amsgrad=True)

            @tf.function
            def _train():
                for x, y in data:
                    optimizer.minimize(
                        lambda: tf.losses.huber_loss(y, model(x), delta=0.01),  # pylint: disable=cell-var-from-loop
                        model.trainable_variables)

            _train()

        # End of _train_fn

        for i in range(100):
            solver.evaluate_and_update_policy(_train_fn)
            if i % 10 == 0:
                conv = pyspiel.exploitability(game, solver.average_policy())
                print("Iteration {} exploitability {}".format(i, conv))
        self.policy = solver.current_policy()
        print(self.policy)


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    for bot in bots:
        bot.train()
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
