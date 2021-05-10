import logging
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from absl import flags, app
import numpy as np
from open_spiel.python.egt.visualization import Dynamics2x2Axes


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")

kuhn_poker_game = pyspiel.load_game('kuhn_poker')


def eval_against_bot(env, q_agent, t_agent, num_episodes):
    """Evaluates `trained_agent` against `test agent` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [q_agent, t_agent]
        else:
            cur_agents = [t_agent, q_agent]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                print(time_step)
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


# kuhn poker
# learn 2 q-learning agents and evaluate the first against the second one (score is of the first agent)
def main(_):
    game = 'kuhn_poker'
    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]

    agents = [tabular_qlearner.QLearner(player_id=0, num_actions=num_actions), tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)]

    # Train the agents
    training_episodes = FLAGS.num_episodes
    for cur_episode in range(training_episodes):
        if cur_episode % int(1e4) == 0:
            win_rates = eval_against_bot(env, agents[0], agents[1], 1000)
            logging.info("Starting episode %s, average score %s", cur_episode, win_rates)
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for i in range(2):
            agents[i].step(time_step)
    # TODO: errort om een reden, check rps qlearner, ttt qlearner
    #for agent in agents:
       # time_step = env.reset()
       # print(agent.step(time_step, True).probs)

if __name__ == "__main__":
    app.run(main)