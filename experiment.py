"""
This file is for the standard training experiments
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
from models.DDPG import DDPG
from models.TD3 import TD3
from models.SAC import SAC

def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("String '{}' is not a known bool value.".format(s))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="./hyperparameters",
                        type=str)
    parser.add_argument("--hyper_param_name",
                        required=True,
                        type=str)
    
    parser.add_argument("--experiment_label",
                        required=True,
                        type=str
                        )
    parser.add_argument("--run_title",
                        required=True,
                        type=str
                        )
    parser.add_argument("--seed",
                        required=False,
                        default=1,
                        type=int
                        )
    
    parser.add_argument("--model",
                        required=True,
                        default="DDPG",
                        type=str
                        )
    # python experiment.py --hyper_param_name Walker --experiment_label test --run_title test --model I_DDPG --env_args forward_reward_weight-0.1,ctrl_cost_weight-100
    """
    Parse the parameters from the command line
    """
    args, unknowns = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknowns[::2], unknowns[1::2])}
    
    """
    Create the directories to hold experiment results
    """
    full_experiment_name = os.path.join(args.experiment_label, args.run_title)
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(os.path.join(full_experiment_name, "hyperparams"))
    
    """
    Read in hyperparameters and save to the experiment folders
    """
    params = utils.get_hyper_parameters(args.hyper_param_directory, args.hyper_param_name)
    params["hyperparams_dir"] = hyperparams_dir
    params["hyper_parameters_name"] = args.hyper_param_name
    params["seed"] = args.seed
    params["model"] = args.model
    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)
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
    Choose whether to run on self constructed env or the guym
    """
    env = gym.make(params["env_name"])
    eval_env = gym.make(params["env_name"])
    params["env"] = env
    params["eval_env"] = env
    utils_for_q_learning.set_random_seed(params)
    """
    Define the model
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Runing on", device)
    
    Constructor = None
    
    if args.model == "DDPG":
        Constructor = DDPG
    elif args.model == "TD3":
        Constructor = TD3
    elif args.model == "SAC":
        Constructor = SAC
    else:
        raise ValueError("Bad module type!")
    
    s0 = env.reset()[0]
    Q_object = Constructor(params = params, 
                           env = env,
                           state_size = len(s0), 
                           action_size = len(env.action_space.low), 
                           device=device)
    Q_target = Constructor(params = params, 
                           env = env,
                           state_size = len(s0), 
                           action_size = len(env.action_space.low), 
                           device=device)
    
    utils_for_q_learning.sync_networks(target=Q_target, online=Q_object, alpha=1, copy=True)
    
    steps = 0

    while steps < params["learning_starts"]:
        s, done, truncated = env.reset()[0], False, False
        episode_steps = 0
        while not (done or truncated):
            a = env.action_space.sample()
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
                        a_eval = Q_object.enact_policy(s_eval, episode, 'test')
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
                
                if np.mean(reward_list) > current_best:
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
    main()
