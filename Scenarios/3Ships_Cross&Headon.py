'''
Scenario: 3Ships_Cross & Head-on
3 ships are in the cross and head-on rules

Definition for data in this scenario:
    ships_num: number of ships, int
    ships_init: initial positions of ships, array[x1,y1; x2, y2; ...]
    ships_goal: target positions of ships, array[x1,y1; x2, y2; ...]
    ships_speed(m/s): (constant) speeds of ships, array[v1; v2; ...]
    ships_head(degree): initial heading angles of ships, array[h1; h2; ...]
    (assume the spaces for all ships are same)
    ship_actions: action spaces of each ship, array[action1, action2, ...]
'''
import numpy as np
ships_num = 3

ships_init = np.zeros((ships_num, 2))
ships_goal = np.zeros((ships_num, 2))
ships_speed = np.zeros((ships_num, 1))
ships_head = np.zeros((ships_num, 1))

ships_init[0, :] = np.array([5000, 0])
ships_goal[0, :] = np.array([5000, 10000])
ships_speed[0] = 20
ships_head[0] = 90

ships_init[1, :] = np.array([0, 5000])
ships_goal[1, :] = np.array([10000, 5000])
ships_speed[1] = 20
ships_head[1] = 0

ships_init[2, :] = np.array([10000, 5000])
ships_goal[2, :] = np.array([0, 5000])
ships_speed[2] = 20
ships_head[2] = 180

# actions of ships
ship_action_space = 1 # heading angle
angle_limit = 2   # heading angle changing range (-2,2)