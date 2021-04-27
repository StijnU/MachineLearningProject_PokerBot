import logging
import pyspiel
import matplotlib.pyplot as plt
import numpy as np
import random
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import random_agent, tabular_qlearner, masked_softmax
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt.dynamics import time_average


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

def plot_trajectory(game, traj, temp):
    payoff_matrix = utils.game_payoffs_array(game)
    boltzmannq = dynamics.MultiPopulationDynamics(payoff_matrix, lambda state, fitness: dynamics.boltzmannq(state, fitness, temp))
    _, ax = plt.subplots(1, 1, subplot_kw=dict(projection="2x2"))
    ax.set_title("Boltzmann Q Learning Dynamic and Trajectory")
    ax.quiver(boltzmannq)
    traj = time_average(traj)
    print(traj)
    ax.plot(traj[:, 0], traj[:, 1])
    plt.savefig('traj.png')
    plt.show()
def main(_):
    game = pyspiel.load_game("matrix_mp")
    # game = pyspiel.create_matrix_game("stag_hunt", "Stag Hunt Game",
    #                                        ["Stag", "Hare"], ["Stag", "Hare"],
    #                                        [[4, 1], [3, 3]], 
    #                                        [[4, 3], [1, 3]])

    discount_factor = 1
    temp = 1

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    q_agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
                tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)]
        
    training_episodes = 5000
    traj = np.zeros((training_episodes, 2))

    """
    # initialize Q-values, as done in https://www.cs.huji.ac.il/~jeff/aamas10/pdf/01%20Full%20Papers/06_01_FP_0040.pdf, p313
    for j in range(2):
        time_step = env.reset()
        state = str(time_step.observations['info_state'][j])
        actions = time_step.observations['legal_actions'][j]
        payoff_matrix = utils.game_payoffs_array(game)
        for a in actions:
            r = min(payoff_matrix[j][a])
            print(r)
            q_agents[j]._q_values[state][a] = 1 / (1 - discount_factor + 0.1) * r
    """
    # Random intialisatie van Q-values
    for j in range(2):
        time_step = env.reset()
        state = str(time_step.observations['info_state'][j])
        actions = time_step.observations['legal_actions'][j]
        for a in actions:
            q_agents[j]._q_values[state][a] = random.uniform(-2, 2)

    for i in range(training_episodes):
        time_step = env.reset()

        chosen_actions = [-1, -1]
        for j in range(2):
            # For each agent:
            #   - lookup current state
            #   - lookup current legal actions
            #   - calculate probabilities for each action, using softmax (boltzmann)
            #   - choose an action with respect to the probability ditribution
            state = str(time_step.observations['info_state'][j])
            actions = time_step.observations['legal_actions'][j]
            probs = masked_softmax.np_masked_softmax(np.array([q_agents[j]._q_values[state][a] for a in actions]) * 1/temp, np.ones(len(actions)))
            chosen_action = random.choices(actions, weights=probs)[0]
            chosen_actions[j] = chosen_action
            q_agents[j]._prev_action = chosen_action
            q_agents[j]._prev_info_state = state
            traj[i, j] = probs[0]

        time_step = env.step(chosen_actions)

        # Episode is over, step all agents with final info state.
        for j in range(2):
            q_agents[j].step(time_step)

    plot_trajectory(game, traj, temp)


if __name__ == "__main__":
    app.run(main)
