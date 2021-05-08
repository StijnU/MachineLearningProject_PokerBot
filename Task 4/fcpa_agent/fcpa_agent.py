#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Provides an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fold_call_bot
import logging
import numpy as np
import pyspiel
import sys
import tensorflow.compat.v1 as tf
import train_fcpa

from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random

logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """
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
        """Initialize an agent to play FCPA poker"""
        pyspiel.Bot.__init__(self)
        self.policy = tf.keras.models.load_model("./fcpa_policy", compile=False)

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = tf.constant(
            state.information_state_tensor(), dtype=tf.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = tf.expand_dims(info_state_vector, axis=0)

        x = info_state_vector
        for layer in self.policy.hidden:
            x = layer(x)
            x = self.policy.activation(x)

        x = self.policy.normalization(x)
        x = self.policy.lastlayer(x)
        x = self.policy.activation(x)
        x = self.policy.out_layer(x)
        x = self.policy.softmax(x)
        x = tf.make_ndarray(tf.make_tensor_proto(x))
        # Only allow player to go all-in if the probability is at least 90%
        if 3 in legal_actions and x[0][3] < 0.9:
            legal_actions = legal_actions[:-1]
            # If player can't go all-in, only allow the player to bet if the probability is at least 50%
            if 2 in legal_actions and x[0][3] < 0.5:
                legal_actions = legal_actions[:-1]
                x[0][1] += x[0][2] + x[0][3]
            else:
                x[0][2] += x[0][3]
        action_prob = {action: x[0][action] for action in legal_actions}
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        # If fold is not an option, add that probability to the call/check action
        if not (0 in legal_actions):
            action_prob[1] += x[0][0]
        # Square and normalize probabilities to make most probable action even more probable
        probs = [prob ** 2 for prob in probs]
        probs = [float(i) / sum(probs) for i in probs]
        # Choose a random action based on their probabilities
        action = np.random.choice(actions, p=probs)
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def LoadAgent(agent_type, game, player_id, rng):
    """Return a bot based on the agent type."""
    seed = 12761381
    if agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "agent":
        return get_agent_for_tournament(player_id)
    elif agent_type == "check_call":
        policy = pyspiel.PreferredActionPolicy([1, 0])
        return pyspiel.make_policy_bot(game, player_id, seed, policy)
    elif agent_type == "fold":
        policy = pyspiel.PreferredActionPolicy([0, 1])
        return pyspiel.make_policy_bot(game, player_id, seed, policy)
    elif agent_type == "50/50":
        return fold_call_bot.FoldCallBot(player_id, rng)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


# Options for bot: "random", "agent", "check_call", "fold", "50/50"
def test_against_bots(bot):
    rng = np.random.RandomState()

    # Make sure poker is compiled into the library, as it requires an optional
    # dependency: the ACPC poker code. To ensure it is compiled in, prepend both
    # the install.sh and build commands with BUILD_WITH_ACPC=ON. See here:
    # https://github.com/deepmind/open_spiel/blob/master/docs/install.md#configuration-conditional-dependencies
    # for more details on optional dependencies.
    games_list = pyspiel.registered_names()
    assert "universal_poker" in games_list

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    print("Creating game: {}".format(fcpa_game_string))
    game = pyspiel.load_game(fcpa_game_string)

    agents = [{
        0: LoadAgent("agent", game, 0, rng),
        1: LoadAgent(bot, game, 1, rng)
    },
        {
        0: LoadAgent(bot, game, 0, rng),
        1: LoadAgent("agent", game, 1, rng)
    }]
    num_rounds = 50000
    utilities = [0, 0]
    wins = [0, 0]
    # Play multiple rounds and take the average result
    for _ in range(2*num_rounds):
        state = game.new_initial_state()
        while not state.is_terminal():
            # The state can be three different types: chance node,
            # simultaneous node, or decision node
            current_player = state.current_player()
            if state.is_chance_node():
                # Chance node: sample an outcome
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = rng.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                # Decision node: sample action for the single current player
                action = agents[0][current_player].step(state)
                state.apply_action(action)

        # Game is now done
        returns = state.returns()
        for pid in range(game.num_players()):
            utilities[pid] += returns[pid]
            if returns[pid] > 0:
                wins[pid] += 1
        agents.reverse()
        utilities.reverse()
        wins.reverse()

    win_rate = [w/2/num_rounds for w in wins]

    print("---------------------------")
    print("Average utility for player {} ({}) is {}".format(0, "agent", utilities[0]/2/num_rounds))
    print("Total utility after {} rounds for player {} ({}) is {}".format(num_rounds*2, 0, "agent", utilities[0]))
    print("Win rate after {} rounds for player {} ({}) is {}".format(num_rounds * 2, 0, "agent", win_rate[0]))
    print("Average utility for player {} ({}) is {}".format(1, bot, utilities[1]/2/num_rounds))
    print("Total utility after {} rounds for player {} ({}) is {}".format(num_rounds*2, 1, bot, utilities[1]))
    print("Win rate after {} rounds for player {} ({}) is {}".format(num_rounds*2, 1, bot, win_rate[1]))
    print("---------------------------")


def main(argv=None):
    train_fcpa.train()
    test_api_calls()
    test_against_bots("random")
    test_against_bots("fold")
    test_against_bots("check_call")
    test_against_bots("50/50")


if __name__ == "__main__":
    sys.exit(main())
