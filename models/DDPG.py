import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from . import utils_for_q_learning
from common.replay_buffer import ReplayBuffer

"""
This file contains the base code for DDPG and TD3.
The differences between DDPG and TD3 is that:
    1. Add aditional noise to the update target in bellman equation (we are now doing this trick to DDPG as well)
    2. Twin value networks in TD3
    3. Delayed update of the policy network, two updates of the value networks correspond to one update of the policy network
we are now doing both 1 and 3 for DDPG, with the only difference between DDPG and TD3 is the twin value networks
"""

class DDPG(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(DDPG, self).__init__()
        print("Twin Critic is off!")
        self.env = env
        self.device = device
        self.params = params
        self.max_a = self.env.action_space.high[0]

        # Defaults to False (standard RBFDQN behavior)

        self.buffer_object = ReplayBuffer(
            size=self.params['max_buffer_size']
        )

        self.state_size, self.action_size = state_size, action_size

        
        self.input_size = self.state_size
        
        self.policy_module = self._make_policy_module()
        self.value_module = self._make_value_module()

        self.criterion = nn.MSELoss()
        
        self.policy_optimizer, self.value_optimizer = self._make_optimizers()
        
        self.num_updates = 0
        self.actor_update_delay = 2
        
        self.to(self.device)

    def _make_policy_module(self):
        module = nn.Sequential(
            nn.Linear(self.input_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.action_size),
            nn.Tanh()
        )
        
        return module
    
    def _make_value_module(self):
        module = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 1)
        )
        
        return module
    
    def _make_optimizers(self):
        policy_optimizer = optim.Adam(self.policy_module.parameters(), lr=self.params["learning_rate_actor"])
        value_optimizer = optim.Adam(self.value_module.parameters(), lr=self.params["learning_rate_critic"])
        return policy_optimizer, value_optimizer
        
    def _get_best_action(self, s):
        assert len(s.shape) == 2
        action_logits = self.policy_module(s)
        return action_logits * self.max_a

    def _get_sa_value(self, s, a):
        assert len(s.shape) == len(a.shape) == 2
        sa_concat = torch.cat([s, a], dim=1) # shape [batch_size, state_size + action_size]
        assert s.shape[0] == sa_concat.shape[0]
        values = self.value_module(sa_concat)
        return values

    def get_best_qvalue_and_action(self, s, add_noise=False):
        best_action = self._get_best_action(s)
        # We do this trick like in TD3.
        # When computing targets for upating, we add noise
        if add_noise:
            noise = torch.clip(torch.normal(0, 0.2, size=best_action.shape).to(self.device), -0.5, 0.5)
            best_action = torch.clip(best_action + noise, -self.max_a, self.max_a)
        value = self._get_sa_value(s, best_action)
        return value, best_action

    def forward(self, s, a):
        '''
		given a batch of s,a , compute Q(s,a) [batch x 1]
		'''
        return self._get_sa_value(s, a)

    def e_greedy_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode,
                                    1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, -1)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.detach().cpu().numpy()[0]
            self.train()
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        assert train_or_test in ["train", "test"]
        self.eval()
        s_matrix = numpy.array(s).reshape(1, -1)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s)
            a = a.cpu().numpy()
        self.train()
        if train_or_test == "train":
            noise = numpy.random.normal(loc=0.,
                                        scale=self.params["noise_std"],
                                        size=len(a))
            a = np.clip(a + noise, -self.max_a, self.max_a)

        return a[0]
    
    def enact_policy(self, s, episode, train_or_test, policy_type="e_greedy"):
        assert policy_type in ["e_greedy","gaussian"], f"Bad policy type: {policy_type}"
        policy_types = {
            'e_greedy': self.e_greedy_policy,
            'gaussian': self.gaussian_policy,
        }

        return policy_types[policy_type](s, episode, train_or_test)
    
    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return {"loss":0, "average_q":0, "average_next_q_max":0}
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix, = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])
        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        with torch.no_grad():
            Q_star, _ = target_Q.get_best_qvalue_and_action(sp_matrix, add_noise=True)
            y = r_matrix + (self.params['gamma'] * (1 - done_matrix) * Q_star.squeeze())
            
        y_hat = self.forward(s_matrix, a_matrix).squeeze()
        loss = self.criterion(y_hat, y)
        self.zero_grad()
        loss.backward()

        self.value_optimizer.step()
        self.zero_grad()

        if self.num_updates % self.actor_update_delay == 0:
            _, best_actions = self.get_best_qvalue_and_action(s_matrix)
            neg_y_hat = -1 * self.forward(s_matrix, best_actions)
            
            neg_y_hat_mean = neg_y_hat.mean()
            neg_y_hat_mean.backward()
            self.policy_optimizer.step()
            self.zero_grad()

            if sync_networks:
                utils_for_q_learning.sync_networks(
                    target=target_Q,
                    online=self,
                    alpha=self.params['target_network_learning_rate'],
                    copy=False)
                
        self.num_updates += 1
        
        loss = loss.item()
        average_q = y_hat.mean().item()
        average_next_q_max = Q_star.mean().item()
        return {
            "loss": loss,
            "average_q": average_q,
            "average_next_q_max": average_next_q_max
        }

class TD3(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(TD3, self).__init__()
        print("Twin Critic is on!")
        self.env = env
        self.device = device
        self.params = params
        self.max_a = self.env.action_space.high[0]

        # Defaults to False (standard RBFDQN behavior)

        self.buffer_object = ReplayBuffer(
            size=self.params['max_buffer_size']
        )

        self.state_size, self.action_size = state_size, action_size

        
        self.input_size = self.state_size
        
        self.policy_module = self._make_policy_module()
        self.value_module_1, self.value_module_2 = self._make_value_module()

        self.criterion = nn.MSELoss()
        
        self.policy_optimizer, self.value_optimizer_1, self.value_optimizer_2 = self._make_optimizers()
        
        self.num_updates = 0
        self.actor_update_delay = 2
        
        self.to(self.device)

    def _make_policy_module(self):
        module = nn.Sequential(
            nn.Linear(self.input_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.action_size),
            nn.Tanh()
        )
        
        return module
    
    def _make_value_module(self):
        module_1 = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 1)
        )
        module_2 = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], self.params["network_size"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size"], 1)
        )
        
        return module_1, module_2
    
    def _make_optimizers(self):
        policy_optimizer = optim.Adam(self.policy_module.parameters(), lr=self.params["learning_rate_actor"])
        value_optimizer_1 = optim.Adam(self.value_module_1.parameters(), lr=self.params["learning_rate_critic"])
        value_optimizer_2 = optim.Adam(self.value_module_2.parameters(), lr=self.params["learning_rate_critic"])
        return policy_optimizer, value_optimizer_1, value_optimizer_2
        
    def _get_best_action(self, s):
        assert len(s.shape) == 2
        action_logits = self.policy_module(s)
        return action_logits * self.max_a

    def _get_sa_value(self, s, a):
        assert len(s.shape) == len(a.shape) == 2
        sa_concat = torch.cat([s, a], dim=1) # shape [batch_size, state_size + action_size]
        assert s.shape[0] == sa_concat.shape[0]
        values_1 = self.value_module_1(sa_concat)
        values_2 = self.value_module_2(sa_concat)
        return values_1, values_2

    def get_best_qvalue_and_action(self, s, add_noise=False):
        best_action = self._get_best_action(s)
        # We do this trick like in TD3.
        # When computing targets for upating, we add noise
        if add_noise:
            noise = torch.clip(torch.normal(0, 0.2, size=best_action.shape).to(self.device), -0.5, 0.5)
            best_action = torch.clip(best_action + noise, -self.max_a, self.max_a)
        values_1, values_2 = self._get_sa_value(s, best_action)
        values = torch.min(torch.cat((values_1, values_2), dim=1), dim=1)[0]
        return values, best_action

    def forward(self, s, a):
        '''
		given a batch of s,a , compute Q(s,a) [batch x 1]
		'''
        return self._get_sa_value(s, a)

    def e_greedy_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode,
                                    1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, -1)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.detach().cpu().numpy()[0]
            self.train()
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        '''
        Given state s, at episode, take random action with p=eps if training
        Note - epsilon is determined by episode
        '''
        '''
		Given state s, at episode, take random action with p=eps if training 
		Note - epsilon is determined by episode
		'''
        assert train_or_test in ["train", "test"]
        self.eval()
        s_matrix = numpy.array(s).reshape(1, -1)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s)
            a = a.cpu().numpy()
        self.train()
        if train_or_test == "train":
            noise = numpy.random.normal(loc=0.,
                                        scale=self.params["noise_std"],
                                        size=len(a))
            a = np.clip(a + noise, -self.max_a, self.max_a)

        return a[0]
    
    def enact_policy(self, s, episode, train_or_test, policy_type="e_greedy"):
        assert policy_type in ["e_greedy","gaussian"], f"Bad policy type: {policy_type}"
        policy_types = {
            'e_greedy': self.e_greedy_policy,
            'gaussian': self.gaussian_policy,
        }

        return policy_types[policy_type](s, episode, train_or_test)
    
    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return {"loss":0, "average_q":0, "average_next_q_max":0}
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix, = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])
        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        with torch.no_grad():
            Q_star, _ = target_Q.get_best_qvalue_and_action(sp_matrix, add_noise=True)
            y = r_matrix + (self.params['gamma'] * (1 - done_matrix) * Q_star.squeeze())

        y_hat_1, y_hat_2 = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat_1.squeeze(), y) + self.criterion(y_hat_2.squeeze(), y)
        self.zero_grad()
        loss.backward()

        self.value_optimizer_1.step()
        self.value_optimizer_2.step()
        self.zero_grad()

        if self.num_updates % self.actor_update_delay == 0:
            _, best_actions = self.get_best_qvalue_and_action(s_matrix)
            neg_y_hat = -1 * self.forward(s_matrix, best_actions)[0]
            
            neg_y_hat_mean = neg_y_hat.mean()
            neg_y_hat_mean.backward()
            self.policy_optimizer.step()
            self.zero_grad()

            if sync_networks:
                utils_for_q_learning.sync_networks(
                    target=target_Q,
                    online=self,
                    alpha=self.params['target_network_learning_rate'],
                    copy=False)
                
        self.num_updates += 1
        
        loss = loss.item()
        average_q = y_hat_1.mean().item()
        average_next_q_max = Q_star.mean().item()
        return {
            "loss": loss,
            "average_q": average_q,
            "average_next_q_max": average_next_q_max
        }