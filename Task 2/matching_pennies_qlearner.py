import logging
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

matching_pennies = pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                            ["Heads", "Tails"], ["Heads", "Tails"],
                                            [[1, -1], [-1, 1]], [[-1, 1], [1, -1]])


def lenient_boltzmann_q(state, fitness, A, temp, kappa):
    y = np.linalg.pinv(A) @ fitness
    if np.linalg.det(A) == 0:
        y += np.full(y.shape, 1/len(y))
    n = len(A)
    u = np.zeros(n)
    for i in range(n):
        for j in range(n):
            u[i] += A[i, j] * y[j] * (sum(y[A[i, :] <= A[i, j]])**kappa - sum(y[A[i, :] < A[i, j]])**kappa) / sum(y[A[i, :] == A[i, j]])
    return dynamics.boltzmannq(state, u, temp)


def matching_pennies_dynamics():
    payoff_matrix = utils.game_payoffs_array(matching_pennies)
    replicator_dynamics = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)
    lenient_dynamics = dynamics.MultiPopulationDynamics(payoff_matrix, lambda state, fitness: lenient_boltzmann_q(state, fitness, payoff_matrix[0], 0.1, 25))
    _, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection="2x2"))
    ax1.streamplot(replicator_dynamics, density=0.6)
    ax1.set_title("Replicator dynamics", fontsize=12)
    ax1.set_xlabel("player 1, probability of choosing heads", fontsize=8.5)
    ax1.set_ylabel("player 2, probability of choosing heads", fontsize=8.5)
    ax2.streamplot(lenient_dynamics, density=0.6)
    ax2.set_title("Lenient boltzmann Q-learning dynamics", fontsize=9)
    ax2.set_xlabel("player 1, probability of choosing heads", fontsize=8.5)
    ax2.set_ylabel("player 2, probability of choosing heads", fontsize=8.5)
    plt.tight_layout()
    plt.show()


def eval_against_bot(env, q_agent, t_agent, num_episodes):
    """Evaluates `trained_agent` against `test agent` for `num_episodes`."""
    score = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        q_agent_output = q_agent.step(time_step, is_evaluation=True)
        t_agent_output = t_agent.step(time_step, is_evaluation=True)
        time_step = env.step([q_agent_output.action, t_agent_output.action])
        score += time_step.rewards[0]
    return score / num_episodes


# matching pennies
# learn 2 q-learning agents and evaluate the first against the second one (score is of the first agent)
def main(_):
    matching_pennies_dynamics()
    env = rl_environment.Environment(matching_pennies)
    num_actions = env.action_spec()["num_actions"]

    agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
              tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)]

    # Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_bot(env, agents[0], agents[1], 1000)
            logging.info("Starting episode %s, average score %s", cur_episode, win_rates)
        time_step = env.reset()
        actions = [0, 0]
        for i in range(2):
            agent_output = agents[i].step(time_step)
            actions[i] = agent_output.action
        time_step = env.step(actions)

        # Episode is over, step all agents with final info state.
        for i in range(2):
            agents[i].step(time_step)
    time_step = env.reset()
    print(agents[1].step(time_step, True).probs)


if __name__ == "__main__":
    app.run(main)
