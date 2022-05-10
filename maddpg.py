"""
Function for building Multi-Agent Deep Deterministic Policy Gradient(MADDPG) algorithm.

Using:
pytroch: 1.10.2
"""
import torch as T
import torch.nn.functional as F
from agent import Agent


class MADDPG:
    def __init__(self, chkpt_dir, actor_dims, critic_dims, n_agents, n_actions,
                 fc1=128, fc2=64, alpha=0.01, beta=0.01):
        """
        :param chkpt_dir: check point directory
        :param actor_dims: number of dimensions for the actor
        :param critic_dims: number of dimensions for the critic
        :param n_agents: number of agents
        :param n_actions: number of actions
        :param fc1: number of dimensions for first layer, default value is 128
        :param fc2: number of dimensions for second layer, default value is 64
        :param alpha: learning rate of actor (target) network, default value is 0.01
        :param beta: learning rate of critic (target) network, default value is 0.01
        """
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx,
                                     fc1=fc1, fc2=fc2, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, exploration=True):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], exploration)
            actions.append(action)
        return actions

    def learn(self, memory):
        """
        agents would learn after filling the bitch size of memory
        :param memory: memory state (from buffer file)
        :return: results after learning
        """

        actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

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
            # actions according to the target network for the new state
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            # actions according to the regular actor network for the current state
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            # actions the agent actually took
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        critic_losses = []
        actor_losses = []
        # handle cost function
        for agent_idx, agent in enumerate(self.agents):
            # current Q estimate
            critic_value = agent.critic.forward(states, old_actions).flatten()
            # target Q value
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            # critic loss
            critic_loss = F.mse_loss(critic_value.float(), target.float())
            # print(critic_loss)
            # critic_loss = F.mse_loss(critic_value, target)
            # critic optimization
            agent.critic.optimizer.zero_grad()
            # critic_loss = critic_loss.detach().requires_grad_(True)
            critic_loss.backward()
            # critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # actor loss
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            # print(actor_loss)
            # actor optimization
            agent.actor.optimizer.zero_grad()
            # actor_loss = actor_loss.detach().requires_grad_(True)
            # actor_loss.backward(retain_graph=True)
            actor_loss.backward()
            agent.actor.optimizer.step()

            agent.update_network_parameters()
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        return critic_losses, actor_losses
