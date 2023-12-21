import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

from . import utils_for_q_learning
from common.replay_buffer import ReplayBuffer

from .DDPG import DDPG, TD3

class Value_Decom_DDPG(DDPG):
    def __init__(self, params, env, state_size, action_size, device):
        # head config is a list of action dimensions
        def Convert(string):
            li = [int(item) for item in string.split("-")]
            return li
        self.head_config = Convert(params["head_config"])
        super().__init__(params, env, state_size, action_size, device)
        
    def _make_policy_module(self):
        module = []
        for i in range(len(self.head_config)):
            sub_module = nn.Sequential(
                nn.Linear(self.input_size, self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.head_config[i]),
                nn.Tanh()
            )
            module.append(sub_module)
        return nn.ModuleList(module) # it returns a bunch of subagents, with each of them in responsible for a certain part of the action dimensions
    
    def _make_value_module(self):
        # We set up attribution model module and critics in this function
        self.attribution_module = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size_factorization"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size_factorization"], len(self.head_config)),
        )
        
        module = []
        for i in range(len(self.head_config)):
            sub_module = nn.Sequential(
                nn.Linear(self.input_size + self.head_config[i], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], 1)
            )
            module.append(sub_module)
        return nn.ModuleList(module)
    
    def _make_optimizers(self):
        params_dic = [{"params":self.attribution_module.parameters(), "lr":self.params["learning_rate_critic"]}]
        self.attribution_optimizer = optim.Adam(params_dic)
        params_dic = [{"params":self.policy_module.parameters(), "lr":self.params["learning_rate_actor"]}]
        policy_optimizer = optim.Adam(params_dic)
        params_dic = [{"params":self.value_module.parameters(), "lr":self.params["learning_rate_critic"]}]
        value_optimizer = optim.Adam(params_dic)
        return policy_optimizer, value_optimizer

    def _get_best_action(self, s):
        assert len(s.shape) == 2 # [batch_size, state_size]
        action_logits = []
        
        for sub_module in self.policy_module:
            action_logits.append(sub_module(s))
        action_logits = torch.concat(action_logits, dim=1).to(self.device)
        return action_logits * self.max_a
    
    def _get_sa_value(self, s, a):  # they should be in shape [batch, state_size] [batch, action_size]
        assert len(s.shape) == len(a.shape) == 2
        batch_size = s.shape[0]
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(a[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
        
        def calculate_values(value_module):
            # breakpoint()
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + decomposed_action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values.sum(dim=1, keepdim=True)
        
        value = calculate_values(self.value_module)
        return value
    
    def get_Q_prediction(self, s, a):
        assert len(s.shape) == len(a.shape) == 2
        batch_size = s.shape[0]
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(a[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
        
        def calculate_values(value_module):
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values
        
        values = calculate_values(self.value_module)
        
        return values
    
    def get_Q_update_target(self, s):
        assert len(s.shape) == 2
        batch_size = s.shape[0]
        
        best_action = self._get_best_action(s)
        noise = torch.clip(torch.normal(0, 0.2, size=best_action.shape).to(self.device), -0.5, 0.5)
        best_action = torch.clip(best_action + noise, -self.max_a, self.max_a)
        # values_1, values_2 = self._get_sa_value(s, best_action)
        
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(best_action[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
            
        def calculate_values(value_module):
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values
        
        values = calculate_values(self.value_module) # [batch_size, num_heads]
        
        return values
        
    
    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return {"loss":0, "average_q":0, "average_next_q_max":0}
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device) # should be [batch, 1]
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        with torch.no_grad():
            # What do we need here?
            # We need the Q_star of each sub critic
            Q_stars = target_Q.get_Q_update_target(sp_matrix) # should be [batch_size, num_heads]
        
        # breakpoint()
        attribution = self.attribution_module(torch.cat((s_matrix, a_matrix), dim=1)).softmax(dim=1) # [batch_size, num_heads]
        attributed_r = r_matrix.unsqueeze(dim=-1) * attribution # [batch_size, num_heads]
        y = attributed_r + (self.params['gamma'] * (1 - done_matrix.unsqueeze(dim=-1)) * Q_stars)
        
        # breakpoint()
        y_hat = self.get_Q_prediction(s_matrix, a_matrix)
        
        loss = self.criterion(y_hat.squeeze(), y)
        
        self.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self.attribution_optimizer.step()
        self.zero_grad()
        
        if self.num_updates % self.actor_update_delay == 0:
            best_actions = self._get_best_action(s_matrix)
            
            neg_y_hat = -1 * self.forward(s_matrix, best_actions)[0]
            
            neg_y_hat_mean = neg_y_hat.mean()
            
            self.zero_grad()
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
        average_next_q_max = Q_stars.sum(dim=1).mean().item()
        return {
            "loss": loss,
            "average_q": average_q,
            "average_next_q_max": average_next_q_max
        }
    

class Value_Decom_TD3(TD3):
    def __init__(self, params, env, state_size, action_size, device):
        # head config is a list of action dimensions
        def Convert(string):
            li = [int(item) for item in string.split("-")]
            return li
        self.head_config = Convert(params["head_config"])
        self.proportion = Convert(params["head_proportion"])
        self.proportion = torch.Tensor(self.proportion).to(device).float().reshape(1, -1)
        self.proportion = (self.proportion / self.proportion.sum(dim=1)).log()
        super().__init__(params, env, state_size, action_size, device)
        
    def _make_policy_module(self):
        module = []
        for i in range(len(self.head_config)):
            sub_module = nn.Sequential(
                nn.Linear(self.input_size, self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.head_config[i]),
                nn.Tanh()
            )
            module.append(sub_module)
        return nn.ModuleList(module) # it returns a bunch of subagents, with each of them in responsible for a certain part of the action dimensions
    
    def _make_value_module(self):
        # We set up attribution model module and critics in this function
        self.attribution_module = nn.Sequential(
            nn.Linear(self.input_size + self.action_size, self.params["network_size_factorization"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
            nn.ReLU(),
            nn.Linear(self.params["network_size_factorization"], len(self.head_config)),
        )
        
        module_1 = []
        for i in range(len(self.head_config)):
            sub_module = nn.Sequential(
                nn.Linear(self.input_size + self.head_config[i], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], 1)
            )
            module_1.append(sub_module)
        module_2 = []
        for i in range(len(self.head_config)):
            sub_module = nn.Sequential(
                nn.Linear(self.input_size + self.head_config[i], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], self.params["network_size_factorization"]),
                nn.ReLU(),
                nn.Linear(self.params["network_size_factorization"], 1)
            )
            module_2.append(sub_module)
        return nn.ModuleList(module_1), nn.ModuleList(module_2)
    
    def _make_optimizers(self):
        params_dic = [{"params":self.attribution_module.parameters(), "lr":self.params["learning_rate_critic"]}]
        self.attribution_optimizer = optim.Adam(params_dic)
        params_dic = [{"params":self.policy_module.parameters(), "lr":self.params["learning_rate_actor"]}]
        policy_optimizer = optim.Adam(params_dic)
        params_dic = [{"params":self.value_module_1.parameters(), "lr":self.params["learning_rate_critic"]}]
        value_optimizer_1 = optim.Adam(params_dic)
        params_dic = [{"params":self.value_module_2.parameters(), "lr":self.params["learning_rate_critic"]}]
        value_optimizer_2 = optim.Adam(params_dic)
        return policy_optimizer, value_optimizer_1, value_optimizer_2

    def _get_best_action(self, s):
        assert len(s.shape) == 2 # [batch_size, state_size]
        action_logits = []
        
        for sub_module in self.policy_module:
            action_logits.append(sub_module(s))
        action_logits = torch.concat(action_logits, dim=1).to(self.device)
        return action_logits * self.max_a
    
    def _get_sa_value(self, s, a):  # they should be in shape [batch, state_size] [batch, action_size]
        assert len(s.shape) == len(a.shape) == 2
        batch_size = s.shape[0]
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(a[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
        
        def calculate_values(value_module):
            # breakpoint()
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + decomposed_action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values.sum(dim=1, keepdim=True)
        
        value_1 = calculate_values(self.value_module_1)
        value_2 = calculate_values(self.value_module_2)
        
        return value_1, value_2
    
    def get_Q_prediction(self, s, a):
        # breakpoint()
        assert len(s.shape) == len(a.shape) == 2
        batch_size = s.shape[0]
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(a[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
        
        def calculate_values(value_module):
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values
        
        values_1 = calculate_values(self.value_module_1)
        values_2 = calculate_values(self.value_module_2)
        attribution = self.attribution_module(torch.cat((s, a), dim=1))
        
        return values_1, values_2, attribution
    
    def get_Q_update_target(self, s):
        assert len(s.shape) == 2
        batch_size = s.shape[0]
        
        best_action = self._get_best_action(s)
        noise = torch.clip(torch.normal(0, 0.2, size=best_action.shape).to(self.device), -0.5, 0.5)
        best_action = torch.clip(best_action + noise, -self.max_a, self.max_a)
        # values_1, values_2 = self._get_sa_value(s, best_action)
        
        action_decomposition = []
        
        action_offset = 0
        for i in range(len(self.head_config)):
            action_decomposition.append(best_action[:, action_offset:action_offset+self.head_config[i]])
            action_offset += self.head_config[i]
            
        def calculate_values(value_module):
            values = torch.zeros(batch_size, len(self.head_config)).to(self.device)
            for i in range(len(action_decomposition)):
                sa_concat = torch.cat([s, action_decomposition[i]], dim=1) # shape [batch_size, state_size + action_size]
                assert s.shape[0] == sa_concat.shape[0]
                values[:, i] = value_module[i](sa_concat).squeeze()
            
            return values
        
        values_1 = calculate_values(self.value_module_1) # [batch_size, num_heads]
        values_2 = calculate_values(self.value_module_2) # [batch_size, num_heads]
        
        # Note that there are two ways to implement this part, one is to find the global minimum
        # one is to find the minimum for each head
        # breakpoint()
        values = torch.min(torch.cat((values_1.unsqueeze(dim=-1), values_2.unsqueeze(dim=-1)), dim=2), dim=2)[0]
        return values
        
    
    def update(self, target_Q, sync_networks=True):
        if len(self.buffer_object) < self.params['batch_size']:
            return {"loss":0, "average_q":0, "average_next_q_max":0}
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device) # should be [batch, 1]
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        with torch.no_grad():
            # What do we need here?
            # We need the Q_star of each sub critic
            Q_stars = target_Q.get_Q_update_target(sp_matrix) # should be [batch_size, num_heads]
        
        # breakpoint()
        attribution = self.attribution_module(torch.cat((s_matrix, a_matrix), dim=1)).softmax(dim=1) # [batch_size, num_heads]
        # breakpoint()
        attributed_r = r_matrix.unsqueeze(dim=-1) * attribution # [batch_size, num_heads]
        y = attributed_r + (self.params['gamma'] * (1 - done_matrix.unsqueeze(dim=-1)) * Q_stars)
        
        y_hat_1, y_hat_2, _ = self.get_Q_prediction(s_matrix, a_matrix)
        # breakpoint()
        loss = self.criterion(y_hat_1, y) + self.criterion(y_hat_2, y)
        
        self.zero_grad()
        loss.backward()
        self.value_optimizer_1.step()
        self.value_optimizer_2.step()
        self.attribution_optimizer.step()
        self.zero_grad()
        
        if self.num_updates % self.actor_update_delay == 0:
            best_actions = self._get_best_action(s_matrix)
            
            neg_y_hat = -1 * self.forward(s_matrix, best_actions)[0]
                
            neg_y_hat_mean = neg_y_hat.mean()
            
            if self.params["proportion_loss"]:
                action_attribution = self.get_Q_prediction(s_matrix, best_actions)[2]
                KL_div = nn.KLDivLoss(reduce="batchmean", log_target=True)
                log_action_attribution = torch.log_softmax(action_attribution, dim=1)
                #sub_values = sub_values / sub_values.sum(dim=1, keepdim=True)
                proportion_loss = KL_div(log_action_attribution, self.proportion)
                # breakpoint()
                proportion_loss_weight = torch.absolute(neg_y_hat_mean.detach()/proportion_loss.detach())/10
                neg_y_hat_mean += proportion_loss_weight*proportion_loss
            
            self.zero_grad()
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
        average_next_q_max = Q_stars.sum(dim=1).mean().item()
        return {
            "loss": loss,
            "average_q": average_q,
            "average_next_q_max": average_next_q_max
        }
