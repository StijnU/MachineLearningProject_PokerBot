"""DQN agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import dqn
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(3e5),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")


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


def main(unused_argv):
  game = "kuhn_poker"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        dqn.DQN(sess, idx, info_state_size, num_actions, hidden_layers_sizes, **kwargs) for idx in range(num_players)
    ]
    evalEpisodes = []
    explList = []
    nashConvList = []
    expl_policies_avg = DQNPolicies(env, agents)

    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        nashConv = exploitability.nash_conv(env.game, expl_policies_avg)
        evalEpisodes.append(ep)
        explList.append(expl)
        nashConvList.append(nashConv)
        logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
        logging.info("[%s] Exploitability AVG %s", ep + 1, nashConv)
        logging.info("_____________________________________________")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)
    plt.plot(evalEpisodes, explList, label="Exploitability")
    plt.plot(evalEpisodes, nashConvList, label="NashConv")
    plt.legend()
    plt.show()


if __name__ == "__main__":
  app.run(main)
