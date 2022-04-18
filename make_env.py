"""
Code for creating a multi-agent environment with one of the scenarios listed in ./scenarios/.

Using:
numpy: 1.21.5
math: Built-in package of Python
Python: 3.9
"""
import Scenarios as Scen
from functions import *


def get_data(scenario_name):
    """
    Get data from the saved file (scenario)
    :param scenario_name:
    :return: data in this given scenario, which includes:
        ships_num: number of ships
        ships_init: initial positions of ships
        ships_term: target positions of ships
        ships_speed: (constant) speeds of ships
        ships_head: initial heading angles of ships
        ship_actions: action spaces of each ship (assume the spaces for all ships are same)
    """
    scenario = Scen.load(scenario_name + ".py")
    return scenario


class MultiAgentEnv:
    """
    Environment for multi agents
    """

    def __init__(self, scenario_name):
        """
        :param scenario_name: the file name, str
            example: scenario_name = '2Ships_Cross'
        """
        self.case = get_data(scenario_name)

        self.ships_num = self.case.ships_num

        self.ships_pos = self.case.ships_init.copy()
        self.ships_speed = self.case.ships_speed.copy()
        self.ships_head = warp_to_360(self.case.ships_head.copy(), self.ships_num)
        self.ships_done_term = [False] * self.ships_num

        self.ships_term = self.case.ships_goal.copy()
        self.ship_action_space = self.case.ship_action_space
        self.angle_limit = self.case.angle_limit
        self.baseline = 1.5
        # ships_obs_space = 2: position
        # ship__obs_space = 3: position + heading
        self.ship_obs_space = 3
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        self.ships_obs_space = []
        for ship_indx in range(self.ships_num):
            self.ships_obs_space.append(self.ship_obs_space)

    def reset(self):
        """
        Function for resetting
        :return: the initial position(state)
        """
        self.ships_pos = self.case.ships_init.copy()
        self.ships_speed = self.case.ships_speed.copy()
        self.ships_head = warp_to_360(self.case.ships_head.copy(), self.ships_num)
        self.ships_done_term = [False] * self.ships_num
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        return self.ships_obs

    def step(self, actions):
        """
        Function for ships to move
        :param actions:
        :return: positions after moving
        """
        for ship_idx in range(self.ships_num):
            if not self.ships_done_term[ship_idx]:
                self.ships_head[ship_idx] += actions[ship_idx] * self.angle_limit
                self.ships_pos[ship_idx, 0] += (self.ships_speed[ship_idx] *
                                                math.cos(math.radians(self.ships_head[ship_idx])))
                self.ships_pos[ship_idx, 1] += (self.ships_speed[ship_idx] *
                                                math.sin(math.radians(self.ships_head[ship_idx])))
        self.ships_head = warp_to_360(self.ships_head, self.ships_num)
        self.ships_obs = np.column_stack((self.ships_pos, self.ships_head))
        return self.ships_obs


if __name__ == '__main__':
    ships = MultiAgentEnv('2Ships_Cross')
    # print(ships.ships_done_term)
    action = [5, 10]
    positions = ships.step(action)
    print(positions)
    # print(len(ships.ship_actions))
    # print(sum(ships.ships_obs_space))
