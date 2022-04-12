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

    chkpt_dir = os.path.dirname(os.path.realpath(__file__))
    maddpg_agents = MADDPG(chkpt_dir, actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01,
                           beta=0.01, scenario=scenario)

    max_size = 1000000
    memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims,
                                    n_agents, n_actions, batch_size=1024)

    dis_redun = 10
    dis_safe = 15
    check_env = CheckState(env.ships_num, env.ships_pos, env.ships_term, env.ships_head, env.ships_speed, dis_redun, dis_safe)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 1000
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        obs[:, 2] = warp_to_360(obs[:, 2], env.ships_num)
        # print(obs)
        # print(obs[:, 2])
        score = 0
        done_reset = False
        done_goal = [False]*n_agents
        episode_step = 0

        while not done_reset:
            actions = maddpg_agents.choose_action(obs)
            # list type: list[array, tpye of array]
            # example: [array([1.9764007], dtype=float32)]
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

            print(reward_term, reward_coll, reward_CORLEG)
            reward = reward_term + reward_coll + reward_CORLEG

            if episode_step >= MAX_STEPS:
                done_reset = True
            # if all(done_goal):
            if any(done_goal):
                done_reset = True
            if done_coll:
                done_reset = True
            memory.store_transition(obs, state, actions, reward, obs_, state_, done_goal)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_.copy()

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
