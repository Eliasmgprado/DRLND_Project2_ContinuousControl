import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import Config, OUNoise, ReplayBuffer
from nn import Actor, Critic

class DDPG_Agent():
    """Deep Deterministic Policy Gradients (DDPG) Actor-Critic Agent."""
    
    def __init__(self, state_size, action_size, random_seed, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            config (obj): Config class object with model configuration.
        """

        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, 
                                 config.fc1_act, config.fc2_act, config.bn).to(config.DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed, 
                                  config.fc1_act, config.fc2_act, config.bn).to(config.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, 
                                   config.fc1_crit, config.fc2_crit, config.bn).to(config.DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed, 
                                    config.fc1_crit, config.fc2_crit, config.bn).to(config.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic, 
                                           weight_decay=config.l2_reg)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = config.noise_decay

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.buffer_size, config.batch_size, random_seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.config.update_every

        if len(self.memory) > self.config.batch_size and self.t_step == 0:
            for _ in range(self.config.update_times):
                experiences = self.memory.sample()
                self.learn(experiences, self.config.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).float().to(self.config.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_decay = self.noise_decay*self.config.noise_decay_factor
            action += np.maximum(self.config.noise_decay, 0.2) * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.clip_grad_crit:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 
                                           self.config.clip_grad_crit) # Clip Gradients
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.clip_grad_act:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 
                                           self.config.clip_grad_act) # Clip Gradients
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ
        
        
        _target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
from nn import ContinuousActorCriticNet      

class PPO_Agent():
    """
    Proximal Policy Optimization (PPO) Actor-Critic Agent 
    based on ShangtongZhang (https://github.com/ShangtongZhang/DeepRL) 
    implementation.
    """
    def __init__(self, state_size, action_size, random_seed, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            random_seed (int): random seed.
            config (obj): Config class object with model configuration.
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = config.epsilon           # Log probability ratio clipping (1-eps; 1+eps)
        self.beta = config.beta                 # Policy Loss entropy regularization coeficient 
        self.add_noise = config.add_noise       # Add OUnoise to actions
        

        self.network = ContinuousActorCriticNet(state_size, action_size, random_seed, 
                                                config.fc1_act, config.fc2_act, 
                                                config.fc1_crit, config.fc2_crit, 
                                                config.bn).to(config.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)
        
        # Initialize enviroment
        env_info = config.env.reset(train_mode=True)[config.brain_name] 
        self.states = env_info.vector_observations 
        self.n_agents = len(env_info.agents)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_decay = config.noise_decay

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def collect_trajectories(self):
        ''' Interact with the enviroment and collect trajectories. '''
        config = self.config
        state_list=[]
        reward_list=[]
        prob_list=[]
        done_list=[]
        
        # Reset enviroment
        env_info = config.env.reset(train_mode=True)[config.brain_name] 
        self.states = env_info.vector_observations 

        states = self.states

        # Collect trajectory
        for t in range(config.trajectory_steps):
            # Predict action based on current state
            pred = self.network(states)
            actions = pred['action'].cpu().detach().numpy()
            
            # Add noise to actions
            if self.add_noise:
                self.noise_decay = self.noise_decay*self.config.noise_decay_factor
                actions += np.maximum(self.config.noise_decay, 0.2) * self.noise.sample()
                
            actions = np.clip(actions, -1, 1) # clip actions

            # Pass actions to the enviroment and collect the response (new_state, rewards, etc.)
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            prob_list.append(pred)
            state_list.append(torch.tensor(states, dtype=torch.float32, device=config.DEVICE))
            reward_list.append(torch.tensor(rewards, dtype=torch.float32, device=config.DEVICE).unsqueeze(-1))
            done_list.append(torch.tensor(dones, dtype=torch.float32, device=config.DEVICE).unsqueeze(-1))

            states = next_states
            
            self.t_step += self.n_agents

        self.states = states
        # Predict actions and state value for last state
        pred = self.network(states)
        prob_list.append(pred)

        return prob_list, state_list, reward_list, done_list
    
    def compute_return_advantage(self, prob_list, state_list, reward_list, done_list):
        ''' Calculate discounted cumulative reward and Generalized Advantage Estimation (GAE).
        
        Config Params
        ======
        config.gamma (float): Return discount.
        config.gae_tau (float): GAE smoothing parameter.
        '''
        config = self.config
        
        advantage_list = [None] * config.trajectory_steps
        return_list = [None] * config.trajectory_steps

        advantages = torch.tensor(np.zeros((self.n_agents, 1)),dtype=torch.float32, device=config.DEVICE)
        returns = prob_list[-1]['v'].detach()
        
        for i in reversed(range(config.trajectory_steps)):
            # Discounted cumullative return (for critic loss)
            returns = reward_list[i] + config.gamma * (1-done_list[i]) * returns
            
            return_list[i] = returns.detach()
            
            # GAE (for actor loss)
            td_error = reward_list[i] + config.gamma * (1-done_list[i]) * prob_list[i + 1]['v'] - prob_list[i]['v']
            advantages = advantages * config.gae_tau * config.gamma * (1-done_list[i]) + td_error

            advantage_list[i] = advantages.detach()
            
        return advantage_list, return_list
    
    def step(self):
        """Collect tragetories and learn.
        
        Config Params
        ======
        config.batch_size (int): Trainig batch size.
        config.SGD_epoch (flt): Number of training epochs (train with same trajectories).
        config.episilon (flt): Log probabiliting clipping.
        config.beta (flt): Policy Loss entropy regularization.
        config.clip_grad (flt): Gradient clipping.
        """
        config = self.config
        # Collect tragetories
        prob_list, state_list, reward_list, done_list = \
            self.collect_trajectories()
        
        # Calculate GAE and discounted cumulative reward
        advantage_list, return_list = self.compute_return_advantage(prob_list, state_list, reward_list, done_list)

        # Create tensors
        state_list = torch.cat(state_list, dim=0).detach()
        advantage_list = torch.cat(advantage_list, dim=0).detach()
        return_list = torch.cat(return_list, dim=0).detach()
        action_list =  torch.cat([prob['action'] for prob in prob_list], dim=0).detach()
        log_prob_list =  torch.cat([prob['log_prob'] for prob in prob_list], dim=0).detach()

        # Normalize GAE
        advantage_list = (advantage_list - advantage_list.mean()) / \
                        (advantage_list.std() + 1e-10) 

        # Create batches for training
        indices = np.asarray(np.random.permutation(np.arange(state_list.size(0))))
        batches = indices[:len(indices) // config.batch_size * config.batch_size].reshape(-1, config.batch_size)
        r = len(indices) % config.batch_size
        if r:
            batches = batches.tolist() + [indices[-r:].tolist()]
        
        # Train the model
        for _ in range(config.SGD_epoch):
            for idxs in batches:
                states_ = state_list[idxs]
                adv_ = advantage_list[idxs]
                ret_ = return_list[idxs]
                act_ = action_list[idxs]
                prob_ = log_prob_list[idxs]
                
                # Get predictions
                pred = self.network(states_, act_)
                
                # Compute log probability ratio
                ratio = (pred['log_prob'] - prob_).exp()
                obj = ratio * adv_
            
                obj_clipped = ratio.clamp(1.0 - self.epsilon,
                                          1.0 + self.epsilon) * adv_
                
                # Compute Losses
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.beta * pred['entropy'].mean()
                value_loss = 0.5 * (ret_- pred['v']).pow(2).mean()
                
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.clip_grad) # Gradient clipping
                self.optimizer.step()
                
        self.epsilon*=.999   # Decreases epsilon (log probability ratio cliping) at each training step.
        self.beta*=.995      # Decreases beta (Policy loss entropy regularization) at each training step.
        
        # Return scores
        return torch.stack(reward_list).squeeze(-1).detach().cpu().data.numpy()
                
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.config.device)
        self.network.eval()
        with torch.no_grad():
            action = self.network(state)['action'].cpu().data.numpy()
        self.network.train()
        if add_noise:
            self.noise_decay = self.noise_decay*self.config.noise_decay_factor
            action += np.maximum(self.config.noise_decay, 0.2) * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Reset action noise."""
        self.noise.reset()