"""
This file is for the ablation experiments. 
"""

import argparse
import os, sys
import datetime
from time import time

import torch
import numpy as np
import gym

from common import utils
from common.logging_utils import MetaLogger

from models import utils_for_q_learning
from models.DDPG import DDPG, TD3
from models.Value_Decom_DDPG import Value_Decom_DDPG, Value_Decom_TD3

from envs.PlainPoint import plainPoint, plainPointMAX
from envs.AntEnv import MyAnt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("String '{}' is not a known bool value.".format(s))


def proportion_loss_experiment(params, Q_object, Q_target, env, eval_env):
    
    full_experiment_name = os.path.join("./test", params["env_name"]+"_"+params["head_proportion"])
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(os.path.join(full_experiment_name, "hyperparams"))
    params["hyperparams_dir"] = hyperparams_dir
    utils.save_hyper_parameters(params, args.seed)
    
    """
    fields to log
    """
    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{args.seed}.pkl"
    meta_logger.add_field("steps", logging_filename)
    meta_logger.add_field("evaluation_reward", logging_filename)
    ### The following fields are used for diagnosis
    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_Q_value", logging_filename)
    
    """
    training
    """
    params["max_steps"] = params["max_steps"]//2
    
    steps = 0
    while steps < params["learning_starts"]:
        s, done, truncated = env.reset()[0], False, False
        episode_steps = 0
        while not (done or truncated):
            a = Q_object.enact_policy(s, 0, 'train', params["policy_type"])
            sp, r, done, truncated, _ = env.step(a)
            Q_object.buffer_object.add(s, a, r, sp, done)
            s = sp
            episode_steps +=1 
            steps += 1
    print("Initialiation Period Finished!")
    
    steps = 0
    episode = 0
    loss_li = []
    Q_li = []
    current_best = float("-inf")
    while steps < params["max_steps"]:
        s, done, truncated = env.reset()[0], False, False
        episode_steps = 0
        episode_reward = 0
        while not (done or truncated):
            a = Q_object.enact_policy(s, episode, 'train', params['policy_type'])
            sp, r, done, truncated, _ = env.step(a)
            Q_object.buffer_object.add(s, a, r, sp, done)
            s = sp
            
            if steps % params['replay_frequency'] == 0:
                statistics_dic = Q_object.update(Q_target, sync_networks=True)
                loss_li.append(statistics_dic["loss"])
                Q_li.append(statistics_dic["average_q"])
            if (steps % params['evaluation_frequency'] == 0) or (steps == params['max_steps'] - 1):
                reward_list = []
                step_list = []
                for _ in range(5):
                    s_eval = eval_env.reset()[0]
                    G_eval, done_eval, t_eval, truncated_eval = 0, False, 0, False
                    while not (done_eval or truncated_eval):
                        a_eval = Q_object.e_greedy_policy(s_eval, episode, 'test')
                        # print(a_eval)
                        sp_eval, r_eval, done_eval, truncated_eval, _ = eval_env.step(a_eval)
                        s_eval, G_eval, t_eval = sp_eval, G_eval + r_eval, t_eval + 1
                    reward_list.append(G_eval)
                    step_list.append(t_eval)
                print("after {} steps, learned policy collects {} average returns in each episode, with average step number as {}".format(steps, np.mean(reward_list), np.mean(step_list)))
                print("average loss {}".format(np.mean(loss_li)))
                print("average Q value {}".format(np.mean(Q_li)))
                meta_logger.append_datapoint("steps", steps, write=True)
                meta_logger.append_datapoint("evaluation_reward", np.mean(reward_list), write=True)
                meta_logger.append_datapoint("average_loss", np.mean(loss_li), write=True)
                meta_logger.append_datapoint("average_Q_value", np.mean(Q_li), write=True)
                
                if steps > params["max_steps"]/2 and np.mean(reward_list) > current_best:
                    print("new best result achieved! Saving the model!")
                    model_path = os.path.join(full_experiment_name, f"seed_{args.seed}.pth")
                    torch.save(Q_object.state_dict(), model_path)
                    current_best = np.mean(reward_list)
                    
                loss_li.clear()
                Q_li.clear()
                print("{:=^50s}".format("Split Line"))
            steps += 1
            episode_steps += 1
            episode_reward += r
        episode += 1
        print("Episode Finishded with Episode steps as {} and Episode rewards as {}".format(episode_steps, episode_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        required=True,
                        type=str)
    parser.add_argument("--head_proportion",
                        required=True,
                        default="",
                        type=str)
    parser.add_argument("--proportion_loss_weight",
                        required=True,
                        default=1,
                        type=float)
    parser.add_argument("--seed",
                        required=False,
                        default=1,
                        type=int)
    
    # Define the model and its corresponding parameters
    args, unknowns = parser.parse_known_args()
    model_path = args.model_path
    
    model_seed = model_path[-10:-4]
    model_dir = model_path[:-10]
    
    hyper_param_dir = os.path.join(model_dir, "hyperparams")
    files = [f for f in os.listdir(hyper_param_dir)]
    
    for f in files:
        if model_seed in f:
            params = utils.get_hyper_parameters(hyper_param_dir, f[:-6])
            break
    params["seed"] = args.seed
    params["proportion_loss"] = True
    params["proportion_loss_weight"] = args.proportion_loss_weight
    params["head_proportion"] = args.head_proportion
    
    # Define the environment
    if params["env_args"]:
            env_args = dict((a.strip(), float(b.strip()))  
                        for a, b in (element.split('-')  
                                    for element in params["env_args"].split('/')))
    else:
        env_args = dict()
        
    if params["env_name"] in ["PlainPoint", "PlainPointMax"]:
        if params["env_name"] == "PlainPoint":
            env = plainPoint(**env_args)
            eval_env = plainPoint(**env_args)
        elif params["env_name"] == "PlainPointMax":
            env = plainPointMAX(**env_args)
            eval_env = plainPointMAX(**env_args)
    else:
        if params["env_name"] == "Ant-v4":
            env = MyAnt()
            eval_env = MyAnt()
        else:
            env = gym.make(params["env_name"], **env_args)
            eval_env = gym.make(params["env_name"], **env_args)
        
    s0 = env.reset()[0]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Runing on", device)
    
    # Define the model
    Constructor = None
        
    if params["model"] == "DDPG":
        if params["using_TD3"]:
            Constructor = TD3
        else:
            Constructor = DDPG
    elif params["model"] == "val_decom":
        if params["using_TD3"]:
            Constructor = Value_Decom_TD3
        else:
            Constructor = Value_Decom_DDPG
    else:
        raise ValueError("Bad module type!")
    
    Q_object = Constructor(params=params,
                           env = env,
                           state_size = len(s0), 
                           action_size = len(env.action_space.low), 
                           device=device)
    Q_target = Constructor(params=params,
                           env = env,
                           state_size = len(s0), 
                           action_size = len(env.action_space.low), 
                           device=device)
    Q_object.load_state_dict(torch.load(model_path, map_location=device))
    Q_target.load_state_dict(torch.load(model_path, map_location=device))
    proportion_loss_experiment(params, Q_object, Q_target, env, eval_env)
    