'''
Code for checking ship states, return reward.

Using:
numpy: 1.21.5
'''
import numpy as np
from make_env import MultiAgentEnv

class CheckState:
    def __init__(self, num_agent, pos_init, pos_term, dis_redun, dis_safe):
        '''
        :param num_agent: number of agents
        :param pos_term: terminal positions
        '''
        self.agents_num = num_agent
        self.pos_init = pos_init
        self.pos_term = pos_term
        self.dis_redun = dis_redun
        self.dis_safe = dis_safe

        distance = []
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i+1, self.agents_num):
                distance.append(np.sqrt((self.pos_init[ship_i, 0]-self.pos_init[ship_j, 0])**2
                                       +(self.pos_init[ship_i, 1]-self.pos_init[ship_j, 1])**2))
        self.dis_cloest = min(distance)
    def check_term(self, state, next_state, done_term):
        '''
        Function for checking destination
        :param state:
        :param next_state:
        :return: reward_term and done_term states
        '''
        reward_term = np.zeros(self.agents_num)
        for ship_idx in range(self.agents_num):
            if not done_term[ship_idx]:
                dis_term = np.sqrt((next_state[ship_idx, 0]-self.pos_term[ship_idx, 0])**2
                                   +(next_state[ship_idx, 1]-self.pos_term[ship_idx, 1])**2)
                if dis_term < self.dis_redun:
                    done_term[ship_idx] = True
                    reward_term[ship_idx] = 100
                else:
                    done_term[ship_idx] = False
                    dis_last = np.sqrt((state[ship_idx, 0]-self.pos_term[ship_idx, 0])**2
                                   +(state[ship_idx, 1]-self.pos_term[ship_idx, 1])**2)
                    reward_term[ship_idx] = dis_last - dis_term
        return reward_term, done_term

    def check_coll(self, state, next_state):
        '''
        Function for checking collision
        :param state:
        :param next_state:
        :return: reward_coll and done_coll states
        '''
        reward_coll = np.zeros(self.agents_num)
        done_coll = False
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i+1, self.agents_num):
                dis_coll = np.sqrt((next_state[ship_i, 0]-next_state[ship_j, 0])**2
                                    +(next_state[ship_i, 1]-next_state[ship_j, 1])**2)
                dis_last = np.sqrt((state[ship_i, 0] - state[ship_j, 0]) ** 2
                                    +(state[ship_i, 1] - state[ship_j, 1]) ** 2)
                if dis_coll < self.dis_cloest:
                    self.dis_cloest = dis_coll

                if dis_coll < self.dis_safe:
                    done_coll = True
                    reward_coll[ship_i] = reward_coll[ship_i] + (dis_coll - dis_last)/20 - 100
                    reward_coll[ship_j] = reward_coll[ship_j] + (dis_coll - dis_last)/20 - 100
                else:
                    reward_coll[ship_i] = reward_coll[ship_i] + (dis_coll - dis_last)/20
                    reward_coll[ship_j] = reward_coll[ship_j] + (dis_coll - dis_last)/20
        return reward_coll, done_coll

if __name__ == '__main__':
    ships = MultiAgentEnv('2Ships_Cross')
    obs = ships.ships_pos
    actions = [5, 10]
    obs_ = ships.step(actions)
    dis_redun = 10
    dis_safe = 15
    check_env = CheckState(ships.ships_num, ships.ships_pos, ships.ships_term, dis_redun, dis_safe)

    done_term = [False] * ships.ships_num
    reward_term, done_term = check_env.check_term(obs, obs_, done_term)
    reward_coll, done_coll = check_env.check_coll(obs, obs_)
    print(reward_term)