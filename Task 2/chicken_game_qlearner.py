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

chicken_game = pyspiel.create_matrix_game("chicken", "Chicken Game",
                                            ["Swerve", "Straight"], ["Swerve", "Straight"],
                                            [[0, -1], [1, -1000]], [[0, 1], [-1, -1000]])


def chicken_dynamics():
    payoff_tensor = utils.game_payoffs_array(chicken_game)
    chicken_dynamic = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)
    ax = plt.subplot(projection="2x2")
    ax.quiver(chicken_dynamic)
    ax.streamplot(chicken_dynamic)
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


# chicken game
# learn 2 q-learning agents and evaluate the first against the second one (score is of the first agent)
def main(_):
    chicken_dynamics()
    env = rl_environment.Environment(chicken_game)
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
