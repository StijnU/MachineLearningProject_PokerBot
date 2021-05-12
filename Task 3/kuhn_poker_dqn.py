"""DQN agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import random
import time
import numpy as np

from open_spiel.python import policy, rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability, masked_softmax
from open_spiel.python.algorithms import dqn
import matplotlib.pyplot as plt


class DQNPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, dqn_policies, temps):
        game = env.game
        player_ids = [0, 1]
        super(DQNPolicies, self).__init__(game, player_ids)
        self._policies = dqn_policies
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}
        self.temps = temps

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)
        p = boltzmann_step(self._policies[cur_player], info_state, self.temps[cur_player], is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def main(unused_argv):

    hidden_layers_sizes = [128]
    num_train_episodes = int(1e4)
    print_every = 100 
    replay_buffer_capacity = int(2e5)

    game = "kuhn_poker"
    num_players = 2

    temps = [1, 1]

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]


    hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,
    }

    with tf.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(sess, idx, info_state_size, num_actions, hidden_layers_sizes, **kwargs) for idx in
            range(num_players)
        ]
        expl_list = []
        nashConv_list = []
        policies = DQNPolicies(env, agents, temps)

        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        for ep in range(num_train_episodes + 1):
            
            expl = exploitability.exploitability(env.game, policies)
            nashConv = exploitability.nash_conv(env.game, policies)
            expl_list.append(expl)
            nashConv_list.append(nashConv)

            if ep % print_every == 0:
              print(80 * "-")
              print("Training Episode: " + str(ep))
              print("Exploitability: " + str(expl_list[-1]))
              print("Time elapsed: " + str(time.time() - start_time))
            
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = boltzmann_step(agents[player_id], time_step, temp=temps[player_id])
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for i in range(2):
                boltzmann_step(agents[i], time_step, temp=temps[i])

        return expl_list, nashConv_list, num_train_episodes


def boltzmann_step(agent, time_step, temp, is_evaluation=False, add_transition_record=True):
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        agent.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][agent.player_id]
      legal_actions = time_step.observations["legal_actions"][agent.player_id]
      action, probs = get_boltzmann_action_and_probabilities(agent, info_state, legal_actions, temp)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      agent._step_counter += 1

      if agent._step_counter % agent._learn_every == 0:
        agent._last_loss_value = agent.learn()

      if agent._step_counter % agent._update_target_network_every == 0:
        agent._session.run(agent._update_target_network)

      if agent._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        agent.add_transition(agent._prev_timestep, agent._prev_action, time_step)

      if time_step.last():  # prepare for the next episode.
        agent._prev_timestep = None
        agent._prev_action = None
        return
      else:
        agent._prev_timestep = time_step
        agent._prev_action = action
    return rl_agent.StepOutput(action=action, probs=probs)


def get_boltzmann_action_and_probabilities(agent, state, actions, temp):
    
    info_state = np.reshape(state, [1, -1])
    q_values = agent._session.run(agent._q_values, feed_dict={agent._info_state_ph: info_state})[0]
    legal_q_values = q_values[actions]

    probs = masked_softmax.np_masked_softmax(legal_q_values * 1/temp, np.ones(len(actions)))
    chosen_action = random.choices(actions, weights=probs)[0]

    return chosen_action, probs


if __name__ == "__main__":
    expl, nash = app.run(main)
