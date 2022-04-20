"""
Main function

CUDA version: 11.2
"""
import os
from functions import *
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from make_env import MultiAgentEnv
from check_state import CheckState

if __name__ == '__main__':
    scenario = '2Ships_Cross'
    env = MultiAgentEnv(scenario)
    n_agents = env.ships_num
    actor_dims = env.ships_obs_space
    critic_dims = sum(actor_dims)

    # action space
    n_actions = env.ship_action_space

    chkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedNetwork'
    maddpg_agents = MADDPG(chkpt_dir, actor_dims, critic_dims, n_agents, n_actions, alpha=0.01, beta=0.01)

    max_size = 1000000
    memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size=1024)

    dis_redun = 10
    dis_safe = 15
    check_env = CheckState(env.ships_num, env.ships_pos, env.ships_term, env.ships_head, env.ships_speed,
                           dis_redun, dis_safe)

    PRINT_INTERVAL = 500
    N_GAMES = 30000
    steps_max = 1000  # can be considered as the simulation time
    steps_exp = N_GAMES / 2
    steps_total = 0

    evaluate = False
    if evaluate:
        maddpg_agents.load_checkpoint()

    score_history = []
    score_best = 0  # for saving path
    score_best_avg = 0  # for saving check points

    path_global = []
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'

    for i in range(N_GAMES + 1):
        obs = env.reset()
        # limits on ships' heading angles
        obs[:, 2] = warp_to_360(obs[:, 2], env.ships_num)
        done_reset = False
        done_goal = [False]*n_agents

        score = 0
        step_episode = 0
        if i < steps_exp:
            Exploration = True
        else:
            Exploration = False

        path_local = []
        path_local.append(obs.reshape(1, -1))

        while not done_reset:
            actions = maddpg_agents.choose_action(obs, Exploration)
            # list type, example: [-1.0, -1.0]
            obs_ = env.step(actions).copy()

            # For local observation problems, observations are not equal to states.
            # For global observation problems, observations are equal to states.
            # For simplification, just set: observations = states.
            state = obs.reshape(1, -1)
            state_ = obs_.reshape(1, -1)

            # reward
            reward_term, done_goal = check_env.check_term(obs, obs_, done_goal)
            env.ships_done_goal = done_goal

            reward_coll, done_coll = check_env.check_coll(obs, obs_)

            reward_CORLEG = check_env.check_CORLEGs(obs, obs_)

            # print(reward_term, reward_coll, reward_CORLEG)
            reward = reward_term + reward_coll + reward_CORLEG

            if step_episode >= steps_max:
                done_reset = True
            # if all(done_goal):
            if any(done_goal):
                done_reset = True
            if done_coll:
                done_reset = True
            memory.store_transition(obs, state, actions, reward, obs_, state_, done_goal)

            if steps_total % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_.copy()
            path_local.append(state)

            score += sum(reward)
            steps_total += 1
            step_episode += 1

        if i == 0:
            score_best = score
            # np.save(result_dir + '/test.npy', path_local)
        elif score > score_best:
            score_best = score
            path_global = path_local

        score_history.append(score)
        score_avg = np.mean(score_history[-100:])
        if not evaluate:
            if score_avg > score_best_avg:
                maddpg_agents.save_checkpoint()
                score_best_avg = score_avg
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(score_avg))

    # save networks
    maddpg_agents.save_checkpoint()
    # save data
    np.save(result_dir + '/score_history.npy', score_history)
    np.save(result_dir + '/path_global.npy', path_global)
