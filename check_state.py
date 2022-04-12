'''
Code for checking ship states, return reward.

Using:
numpy: 1.21.5
'''
from functions import *
from make_env import MultiAgentEnv

class CheckState:
    def __init__(self, num_agent, pos_init, pos_term, head_init, vel_init, dis_redun, dis_safe):
        '''
        :param num_agent: number of agents
        :param pos_init: initial positions of ships
        :param pos_term: terminal positions of ships
        :param head_init: initial heading angles of ships
        :param vel_init: initial velocities of ships
        :param dis_redun: redundant distance
        :param dis_safe: safe distance
        '''
        self.agents_num = num_agent
        self.pos_init = pos_init
        self.pos_term = pos_term
        self.heads = head_init
        self.speeds = vel_init
        self.dis_redun = dis_redun
        self.dis_safe = dis_safe

        distance = []
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i+1, self.agents_num):
                distance.append(euc_dist(self.pos_init[ship_i, 0], self.pos_init[ship_j, 0],
                                         self.pos_init[ship_i, 1], self.pos_init[ship_j, 1]))
        self.dis_cloest = min(distance)

        self.rules_list = ['Null'] * self.agents_num * self.agents_num
        self.rules_table = np.array(self.rules_list).reshape(self.agents_num, self.agents_num)
        for ship_i in range(self.agents_num):
            for ship_j in range(self.agents_num):
                self.rules_table[ship_i, ship_j] = 'Null'

    def check_term(self, state, next_state, done_term):
        '''
        Function for checking destination
        :param state:
        :param next_state:
        :return: reward_term and done_term states
                 reward_term: reward according to terminal states
        '''
        reward_term = np.zeros(self.agents_num)
        for ship_idx in range(self.agents_num):
            if not done_term[ship_idx]:
                dis_term = euc_dist(next_state[ship_idx, 0], self.pos_term[ship_idx, 0],
                                    next_state[ship_idx, 1], self.pos_term[ship_idx, 1])
                if dis_term < self.dis_redun:
                    done_term[ship_idx] = True
                    reward_term[ship_idx] = 100
                else:
                    done_term[ship_idx] = False
                    dis_last = euc_dist(state[ship_idx, 0], self.pos_term[ship_idx, 0],
                                        state[ship_idx, 1], self.pos_term[ship_idx, 1])
                    reward_term[ship_idx] = dis_last - dis_term
        return reward_term, done_term

    def check_coll(self, state, next_state):
        '''
        Function for checking collision
        :param state:
        :param next_state:
        :return: reward_coll and done_coll states
                 reward_coll: reward according to collision states
        '''
        reward_coll = np.zeros(self.agents_num)
        done_coll = False
        for ship_i in range(self.agents_num):
            for ship_j in range(ship_i+1, self.agents_num):
                dis_coll = euc_dist(next_state[ship_i, 0], next_state[ship_j, 0],
                                    next_state[ship_i, 1], next_state[ship_j, 1])
                dis_last = euc_dist(state[ship_i, 0], state[ship_j, 0],
                                    state[ship_i, 1], state[ship_j, 1])
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

    def check_CORLEGs(self, state, next_state):
        '''
        :param state:
        :param next_state:
        :return: reward_CORLEGs: reward according to CORLEGs states
        '''
        pos = state[:, 0:2]
        head = state[:, 2]
        # pos_ = next_state[:, 0:1]
        head_ = next_state[:, 2]
        head_diff = warp_to_180(head_ - head, self.agents_num)

        reward_CORLEGs = np.zeros(self.agents_num)
        for ship_i in range(self.agents_num):
            for ship_j in range(self.agents_num):
                # update the CORLEGs table
                if ship_i == ship_j:
                    self.rules_table[ship_i, ship_j] = 'Null'
                else:
                    self.rules_table[ship_i, ship_j] = colregs_rule(
                        pos[ship_i, 0], pos[ship_i, 1],
                        head[ship_i], self.speeds[ship_i],
                        pos[ship_j, 0], pos[ship_j, 1],
                        head[ship_j], self.speeds[ship_j])
                    # get reward according heading angles
                    if self.rules_table[ship_i, ship_j] == 'HO-GW' or 'OT-GW' or 'CR-GW':
                        reward_CORLEGs[ship_i] += head_diff[ship_i]
                    if self.rules_table[ship_i, ship_j] == 'OT-SO' or 'CR-SO':
                        if head_diff[ship_i] == 0:
                            reward_CORLEGs[ship_i] += 5
                        else:
                            reward_CORLEGs[ship_i] -= abs(head_diff[ship_i])
        return reward_CORLEGs


if __name__ == '__main__':
    ships = MultiAgentEnv('2Ships_Cross')
    # obs = ships.ships_pos
    # actions = [5, 10]
    # obs_ = ships.step(actions)
    dis_redun = 10
    dis_safe = 15
    check_env = CheckState(ships.ships_num, ships.ships_pos, ships.ships_term, ships.ships_head, ships.ships_speed, dis_redun, dis_safe)
    #
    # done_term = [False] * ships.ships_num
    # reward_term, done_term = check_env.check_term(obs, obs_, done_term)
    # reward_coll, done_coll = check_env.check_coll(obs, obs_)
    print(check_env.rules_table)