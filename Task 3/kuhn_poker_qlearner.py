import logging
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment, policy
from open_spiel.python.algorithms import tabular_qlearner, random_agent, exploitability
from open_spiel.python.egt import utils
from absl import flags, app
import numpy as np


FLAGS = flags.FLAGS

class QPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, q_policies):
        game = env.game
        player_ids = [0, 1]
        super(QPolicies, self).__init__(game, player_ids)
        self._policies = q_policies
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

# kuhn poker
# learn 2 q-learning agents and evaluate the first against the second one (score is of the first agent)
def main(_):
    game = 'kuhn_poker'
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    

    agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions), tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)]
    
    policies = QPolicies(env, agents)

    expl = []
    conv = []
    # Train the agents
    training_episodes = 10000
    for cur_episode in range(training_episodes):
        current_expl = exploitability.exploitability(env.game, policies)
        current_conv = exploitability.nash_conv(env.game, policies)
        expl.append(current_expl)
        conv.append(current_conv)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for i in range(2):
            agents[i].step(time_step)
    
    return expl, conv, training_episodes

    # TODO: errort om een reden, check rps qlearner, ttt qlearner
    #for agent in agents:
       # time_step = env.reset()

if __name__ == "__main__":
    app.run(main)