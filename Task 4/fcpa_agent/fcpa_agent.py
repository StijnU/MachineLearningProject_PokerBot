#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import logging
import numpy as np
import pyspiel
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import evaluate_bots, nfsp
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from open_spiel.python.bots import uniform_random

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


def train():
    """Trains the model and save the trained policy"""
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    num_players = 2
    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in [128, 32]]
    kwargs = {
        "replay_buffer_capacity": int(5e3),
        "epsilon_decay_duration": int(3e3),
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    with tf.Session() as sess:
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                      reservoir_buffer_capacity=int(5e3), anticipatory_param=0.1,
                      **kwargs) for idx in range(num_players)
        ]
        learned_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
        sess.run(tf.global_variables_initializer())

        for ep in range(int(3e3)):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
        # for agent in agents:
        #     agent.save("/tmp/fcpa_checkpoints")
    return learned_policy


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)

        # # Reload policy
        # fcpa_game_string = pyspiel.hunl_game_string("fcpa")
        # game = pyspiel.load_game(fcpa_game_string)
        # num_players = 2
        # env_configs = {"players": num_players}
        # env = rl_environment.Environment(game, **env_configs)
        # info_state_size = env.observation_spec()["info_state"][0]
        # num_actions = env.action_spec()["num_actions"]
        #
        # hidden_layers_sizes = [int(l) for l in [128, 32]]
        # kwargs = {
        #     "replay_buffer_capacity": int(5e3),
        #     "epsilon_decay_duration": int(3e2),
        #     "epsilon_start": 0.06,
        #     "epsilon_end": 0.001,
        # }
        #
        # with tf.Session() as sess:
        #     agents = [
        #         nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
        #                   reservoir_buffer_capacity=int(5e3), anticipatory_param=0.1,
        #                   **kwargs) for idx in range(num_players)
        #     ]
        #
        #     sess.run(tf.global_variables_initializer())
        #
        #     for agent in agents:
        #         print(agent.has_checkpoint("/tmp/fcpa_checkpoints"))
        #         agent.restore("/tmp/fcpa_checkpoints")
        #     learned_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
        self.policy = None
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
        action_prob = self.policy.action_probabilities(state, self.player_id)
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        action = np.random.choice(actions, p=probs)
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    learned_policy = train()
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    for bot in bots:
        bot.policy = learned_policy
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._policies[cur_player]._session = sess
            with self._policies[cur_player].temp_mode_as(self._mode):
                p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
            prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "agent", "check_call", "fold"
flags.DEFINE_string("player0", "agent", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "check_call", "Type of the agent for player 1.")


def LoadAgent(agent_type, game, player_id, rng):
    """Return a bot based on the agent type."""
    if agent_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif agent_type == "agent":
        learned_policy = train()
        bot = get_agent_for_tournament(player_id)
        bot.policy = learned_policy
        return bot
    elif agent_type == "check_call":
        policy = pyspiel.PreferredActionPolicy([1, 0])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    elif agent_type == "fold":
        policy = pyspiel.PreferredActionPolicy([0, 1])
        return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def test_against_bots(_):
    rng = np.random.RandomState(FLAGS.seed)

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

    agents = [
        LoadAgent(FLAGS.player0, game, 0, rng),
        LoadAgent(FLAGS.player1, game, 1, rng)
    ]
    num_rounds = 10
    utilities = [0, 0]
    # Play multiple rounds and take the average result
    for _ in range(num_rounds):
        state = game.new_initial_state()

        # Print the initial state
        print("INITIAL STATE")
        print(str(state))

        while not state.is_terminal():
            # The state can be three different types: chance node,
            # simultaneous node, or decision node
            current_player = state.current_player()
            if state.is_chance_node():
                # Chance node: sample an outcome
                outcomes = state.chance_outcomes()
                num_actions = len(outcomes)
                print("Chance node with " + str(num_actions) + " outcomes")
                action_list, prob_list = zip(*outcomes)
                action = rng.choice(action_list, p=prob_list)
                print("Sampled outcome: ",
                      state.action_to_string(state.current_player(), action))
                state.apply_action(action)
            else:
                # Decision node: sample action for the single current player
                legal_actions = state.legal_actions()
                for action in legal_actions:
                    print("Legal action: {} ({})".format(
                        state.action_to_string(current_player, action), action))
                action = agents[current_player].step(state)
                action_string = state.action_to_string(current_player, action)
                print("Player ", current_player, ", chose action: ",
                      action_string)
                state.apply_action(action)

            print("")
            print("NEXT STATE:")
            print(str(state))

        # Game is now done. Print utilities for each player
        returns = state.returns()
        for pid in range(game.num_players()):
            print("Utility for player {} is {}".format(pid, returns[pid]))
            utilities[pid] += returns[pid]
        utilities.reverse()
        agents.reverse()
    print("---------------------------")
    print("Average utility for player {} ({}) is {}".format(0, FLAGS.player0, utilities[0]/num_rounds))
    print("Average utility for player {} ({}) is {}".format(1, FLAGS.player1, utilities[1]/num_rounds))


def main(argv=None):
    test_api_calls()
    app.run(test_against_bots)


if __name__ == "__main__":
    sys.exit(main())
