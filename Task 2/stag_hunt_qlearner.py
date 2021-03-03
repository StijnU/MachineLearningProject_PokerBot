import logging
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
from open_spiel.python.egt.visualization import Dynamics2x2Axes

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

stag_hunt_game = pyspiel.create_matrix_game("stag_hunt", "Stag Hunt Game",
                                            ["Stag", "Hare"], ["Stag", "Hare"],
                                            [[4, 1], [3, 2]], [[4, 3], [1, 2]])


def stag_hunt_dynamics():
    payoff_tensor = utils.game_payoffs_array(stag_hunt_game)
    stag_hunt_dynamic = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    fig = plt.Figure()
    Dynamics2x2Axes(fig, [0, 0, 4, 4]).streamplot(stag_hunt_dynamic)  # TODO: not working, does not show figure
    plt.show()


def eval_against_random_bot(env, q_agent, r_agent, num_episodes):
    """Evaluates `trained_agent` against `random_agent` for `num_episodes`."""
    score = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        q_agent_output = q_agent.step(time_step, is_evaluation=True)
        r_agent_output = r_agent.step(time_step, is_evaluation=True)
        time_step = env.step([q_agent_output.action, r_agent_output.action])
        score += time_step.rewards[0]
    return score / num_episodes


# stag hunt game
def main(_):
    stag_hunt_dynamics()
    env = rl_environment.Environment(stag_hunt_game)
    num_actions = env.action_spec()["num_actions"]

    q_agent = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions)

    # random agents for evaluation
    r_agent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

    # Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_random_bot(env, q_agent, r_agent, 1000)
            logging.info("Starting episode %s, average score %s", cur_episode, win_rates)
        time_step = env.reset()
        q_agent_output = q_agent.step(time_step)
        time_step = env.step([q_agent_output.action, r_agent.step(time_step).action])

        # Episode is over, step all agents with final info state.
        q_agent.step(time_step)


if __name__ == "__main__":
    app.run(main)
