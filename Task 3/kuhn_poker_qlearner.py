import logging
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment, policy
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import tabular_qlearner, masked_softmax, exploitability
from open_spiel.python.egt import utils
from absl import flags, app
import numpy as np
import random
import time


FLAGS = flags.FLAGS

class QPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, q_policies, temps):
        game = env.game
        player_ids = [0, 1]
        self.temps = temps
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
        p = boltzmann_step(self._policies[cur_player], info_state, self.temps[cur_player], is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

# kuhn poker
# learn 2 q-learning agents and evaluate the first against the second one (score is of the first agent)
def train_Q_learning_agents(training_episodes, eval_every, step_size, discount_factor, temps):
    game = 'kuhn_poker'
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions, step_size=step_size, discount_factor=discount_factor), tabular_qlearner.QLearner(player_id=1, num_actions=num_actions, step_size=step_size, discount_factor=discount_factor)]
    policies = QPolicies(env, agents, temps)

    expl = []
    conv = []
    # Train the agents
    start_time = time.time()
    for cur_episode in range(training_episodes + 1):
        
        if cur_episode % eval_every == 0:
            current_expl = exploitability.exploitability(env.game, policies)
            current_conv = exploitability.nash_conv(env.game, policies)
            expl.append(current_expl)
            conv.append(current_conv)
            print(80 * "-")
            print("Training Episode: " + str(cur_episode))
            print("Exploitability: " + str(expl[-1]))
            print("Time elapsed: " + str(time.time() - start_time))

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = boltzmann_step(agents[player_id], time_step, temp=temps[player_id])
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for i in range(2):
            boltzmann_step(agents[i], time_step, temp=temps[i])
    
    return expl, conv

def boltzmann_step(agent, time_step, temp, is_evaluation=False):
    info_state = str(time_step.observations["info_state"][agent._player_id])
    legal_actions = time_step.observations["legal_actions"][agent._player_id]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    # Act step: don't act at terminal states.
    if not time_step.last():
        action, probs = get_boltzmann_action_and_probabilities(agent, info_state, legal_actions, temp)

    # Learn step: don't learn during evaluation or at first agent steps.
    if agent._prev_info_state and not is_evaluation:
      target = time_step.rewards[agent._player_id]
      if not time_step.last():  # Q values are zero for terminal.
        target += agent._discount_factor * max(
            [agent._q_values[info_state][a] for a in legal_actions])

      prev_q_value = agent._q_values[agent._prev_info_state][agent._prev_action]
      agent._last_loss_value = target - prev_q_value
      agent._q_values[agent._prev_info_state][agent._prev_action] += (
          agent._step_size * agent._last_loss_value)

      if time_step.last():  # prepare for the next episode.
        agent._prev_info_state = None
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      agent._prev_info_state = info_state
      agent._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

def get_boltzmann_action_and_probabilities(agent, state, actions, temp):
    probs = masked_softmax.np_masked_softmax(np.array([agent._q_values[state][a] for a in actions]) * 1/temp, np.ones(len(actions)))
    chosen_action = random.choices(actions, weights=probs)[0]
    return chosen_action, probs

if __name__ == "__main__":
    app.run(main)