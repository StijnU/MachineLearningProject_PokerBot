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

if __name__ == "__main__":
    app.run(main)