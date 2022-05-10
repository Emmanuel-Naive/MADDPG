"""
Function for building Multi-Agent Deep Deterministic Policy Gradient(MADDPG) algorithm.
Using:
pytroch: 1.10.2
"""
import torch
import torch as T
import torch.nn.functional as F
from agent import Agent


class MADDPG:
    def __init__(self, chkpt_dir, actor_dims, critic_dims, n_agents, n_actions,
                 scenario, alpha=0.01, beta=0.01, fc1=64, fc2=64,
                 gamma=0.99, tau=0.01):
        """
        :param chkpt_dir: check point directory
        :param actor_dims: number of dimensions for the actor
        :param critic_dims: number of dimensions for the critic
        :param n_agents: number of agents
        :param n_actions: number of actions
        :param scenario: name of scenario
        :param alpha: learning rate of actor (target) network, default value is 0.01
        :param beta: learning rate of critic (target) network, default value is 0.01
        :param fc1: number of dimensions for first layer, default value is 64
        :param fc2: number of dimensions for second layer, default value is 64
        :param gamma: discount factor, default value is 0.95
        :param tau: learning rate for adam optimization,  default value is 0.01
        """
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += '\SavedNetwork'
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        """
        agents would learn after filling the bitch size of memory
        :param memory: memory state (from buffer file)
        :return: results after learning
        """
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        # all these three different actions are needed to calculate the loss function
        all_agents_new_actions = []  # actions according to the target network for the new state
        all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
        old_agents_actions = []  # actions the agent actually took

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            # new actions
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        # handle cost function
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            # print('critic_value_:',critic_value_.shape,critic_value_)
            critic_value = agent.critic.forward(states, old_actions).flatten()
            # print('critic_value:',critic_value.shape,critic_value)
            # print('rewards[:,agent_idx]',rewards[:,agent_idx].shape,rewards[:,agent_idx])
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            # print('target:',target.shape,target)
            critic_loss = F.mse_loss(critic_value, target)
            # print('critic_loss:',critic_loss.shape,critic_loss)
            agent.critic.optimizer.zero_grad()
            critic_loss = T.tensor(critic_loss.clone(), dtype=T.float, requires_grad=True).to(device)
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
