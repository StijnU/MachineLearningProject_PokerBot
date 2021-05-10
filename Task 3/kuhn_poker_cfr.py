# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import matplotlib.pyplot as plt


def main(_):
    game = pyspiel.load_game("kuhn_poker")

    cfr_solver = cfr.CFRSolver(game)
    iterations = 1000

    expl_data = []
    conv_data = []

    for i in range(iterations):
        cfr_solver.evaluate_and_update_policy()
        current_expl = exploitability.exploitability(game, cfr_solver.average_policy())
        current_conv = exploitability.nash_conv(game, cfr_solver.average_policy())
        expl_data.append(current_expl)
        conv_data.append(current_conv)

    print(cfr_solver.average_policy().action_probability_array)
    return expl_data, conv_data, range(iterations)

if __name__ == "__main__":
    app.run(main)
