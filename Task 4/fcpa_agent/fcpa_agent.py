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
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import evaluate_bots, rcfr, dqn
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


def train():
    """Trains the model and save the trained policy"""
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    num_players = 2
    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in [128, ]]
    kwargs = {
        "replay_buffer_capacity": int(2e2),
        "epsilon_decay_duration": int(3e3),
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    with tf.Session() as sess:
        agents = [
            dqn.DQN(sess, idx, info_state_size, num_actions, hidden_layers_sizes, **kwargs) for idx in
            range(num_players)
        ]
        expl_policies_avg = DQNPolicies(env, agents)

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
        for agent in agents:
            agent.save("./checkpoints")


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)

        fcpa_game_string = pyspiel.hunl_game_string("fcpa")
        game = pyspiel.load_game(fcpa_game_string)
        num_players = 2
        env_configs = {"players": num_players}
        self.env = rl_environment.Environment(game, **env_configs)
        info_state_size = self.env.observation_spec()["info_state"][0]
        num_actions = self.env.action_spec()["num_actions"]

        hidden_layers_sizes = [int(l) for l in [128, ]]
        kwargs = {
            "replay_buffer_capacity": int(2e2),
            "epsilon_decay_duration": int(3e3),
            "epsilon_start": 0.06,
            "epsilon_end": 0.001,
        }
        with tf.Session() as sess:
            # pylint: disable=g-complex-comprehension
            self.agent = dqn.DQN(sess, player_id, info_state_size, num_actions, hidden_layers_sizes, **kwargs)

        self.player_id = player_id
        self.state = None
        self.previousAction = [None, None]

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        self.env.set_state(state)

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        self.env.set_state(state)
        self.previousAction[player_id] = action

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        with tf.Session() as sess:
            self.agent._session = sess
            if len(state.legal_actions()) == 0:
                return pyspiel.INVALID_ACTION
            action = self.agent.step(time_step=self.env.get_time_step(), is_evaluation=True)
            self.previousAction[self.player_id] = action
        return action

    def reload_policy(self):
        if self.agent.has_checkpoint("./checkpoints"):
            self.agent.restore("./checkpoints")


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """

    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    train()
    for bot in bots:
        bot.reload_policy()
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


class DQNPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, dqn_policies):
        game = env.game
        player_ids = [0, 1]
        super(DQNPolicies, self).__init__(game, player_ids)
        self._policies = dqn_policies
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
        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
