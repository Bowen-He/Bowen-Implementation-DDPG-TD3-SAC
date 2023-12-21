"""
This file will iterate from a start directory, finding all files ending with .pth, namely the models. It then will read in those models and generate videos under the corresponding directories.
"""

import gym
import torch
import torch.nn as nn
from common import utils
from models.DDPG import DDPG, TD3
from models.Value_Decom_DDPG import Value_Decom_DDPG, Value_Decom_TD3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import argparse
from pathlib import Path
import os

from envs.PlainPoint import plainPoint, plainPointMAX
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def get_data_for_humanoid(data):
        """
        Given a MuJoCO data, get the things of interest.
        """
        mass_point = data.xpos[1].copy()
        right_foot_index = 8
        left_foot_index = 11
        indices_ground = np.where(data.contact.geom1 == 0)[0]
        indices_right_foot = np.where(data.contact.geom2 == right_foot_index)[0]
        indices_left_foot = np.where(data.contact.geom2 == left_foot_index)[0]
        
        common_indices_right_foot = np.intersect1d(indices_ground, indices_right_foot)
        common_indices_left_foot = np.intersect1d(indices_ground, indices_left_foot)
        
        pos_right_foot = data.contact.pos[common_indices_right_foot]
        pos_left_foot = data.contact.pos[common_indices_left_foot]
        
        return mass_point.squeeze(), pos_right_foot.squeeze(), pos_left_foot.squeeze()

def get_data_for_ant(data):
    mass_point = data.xpos[1].copy()
    front_left_leg_index = 4
    front_right_leg_index = 7
    back_left_leg_index = 10
    back_right_leg_index = 13
    
    indices_ground = np.where(data.contact.geom1 == 0)[0]
    indices_front_left_leg = np.where(data.contact.geom2 == front_left_leg_index)[0]
    indices_front_right_leg = np.where(data.contact.geom2 == front_right_leg_index)[0]
    indices_back_left_leg = np.where(data.contact.geom2 == back_left_leg_index)[0]
    indices_back_right_leg = np.where(data.contact.geom2 == back_right_leg_index)[0]
    
    common_indices_front_left_leg = np.intersect1d(indices_ground, indices_front_left_leg)
    common_indices_front_right_leg = np.intersect1d(indices_ground, indices_front_right_leg)
    common_indices_back_left_leg = np.intersect1d(indices_ground, indices_back_left_leg)
    common_indices_back_right_leg = np.intersect1d(indices_ground, indices_back_right_leg)
    
    pos_front_left_leg = data.contact.pos[common_indices_front_left_leg]
    pos_front_right_leg = data.contact.pos[common_indices_front_right_leg]
    pos_back_left_leg = data.contact.pos[common_indices_back_left_leg]
    pos_back_right_leg = data.contact.pos[common_indices_back_right_leg]
    
    return mass_point.squeeze(), pos_front_left_leg.squeeze(), pos_front_right_leg.squeeze(), pos_back_left_leg.squeeze(), pos_back_right_leg.squeeze()

def get_data_for_halfcheetah(data):
    mass_point = data.xpos[1].copy()
    front_foot_index = 8
    back_foot_index = 5
    indices_ground = np.where(data.contact.geom1 == 0)[0]
    indices_front_foot = np.where(data.contact.geom2 == front_foot_index)[0]
    indices_back_foot = np.where(data.contact.geom2 == back_foot_index)[0]
    
    common_indices_front_foot = np.intersect1d(indices_ground, indices_front_foot)
    common_indices_back_foot = np.intersect1d(indices_ground, indices_back_foot)
    
    pos_front_foot = data.contact.pos[common_indices_front_foot]
    pos_back_foot = data.contact.pos[common_indices_back_foot]
    # breakpoint()
    
    return mass_point.squeeze(), pos_front_foot.squeeze(), pos_back_foot.squeeze()
    
def get_data_for_none(data):
    return

def generate_a_video_for_value_array(value_array, video_path, labels=None, moving_average=1):
        """
        Generate a video from a value array.

        Args:
            value_array (np.array): The value array containing data, should be in shape [num_step, num_head]
            video_path (string): The path to generate the video.
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_recorder = cv2.VideoWriter(video_path, fourcc, 25, (480, 480))
        
        if labels is None:
            labels = []
            for i in range(value_array.shape[0]):
                labels.append(f"sub module {i}")
        
        count = 0
        for i in range(value_array.shape[0]):
            sub_array = value_array[:i+1, :]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            for j in range(sub_array.shape[1]):
                sub_values = sub_array[:, j]
                sub_values = np.convolve(sub_values, np.ones(moving_average)/moving_average, mode='valid')
                ax.plot(sub_values, label=labels[j])
            
            ax.legend()
            img = get_img_from_fig(fig)
            img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
            plt.close("all")

            video_recorder.write(img)
            
            count += 1
            
        video_recorder.release()

def generate_a_video_for_frame_array(frame_array, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_recorder = cv2.VideoWriter(video_path, fourcc, 25, (480, 480))
        for i, frame in enumerate(frame_array):
            video_recorder.write(frame)
            
        video_recorder.release()
        
def generate_a_video_for_gait_analysis(data_array, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_recorder = cv2.VideoWriter(video_path, fourcc, 25, (480, 480))
    color = ["red", "blue", "black", "purple"]
    contact_counts = np.zeros(len(data_array[0]) - 1)
    
    count = 0
    for i in data_array:
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        
        for num_point, contact_point in enumerate(i[1:]):
            if contact_point.size != 0:
                point = contact_point[:2] - i[0][:2]
                plt.scatter(x=point[0], y=point[1], c=color[num_point])
                contact_counts[num_point] += 1
        img = get_img_from_fig(fig)
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
        plt.close("all")
        video_recorder.write(img)
        
        count += 1
    print(contact_counts)
    video_recorder.release()
    
def generate_videos_for_value_decomposition_TD3(env, model, video_directory):
    
    def get_critic_values(policy: torch.nn.Module, state: np.array, action: np.array):
        s_matrix = np.array(state).reshape(1, -1) # should be in the shape [1, state_size]
        a_matrix = np.array(action).reshape(1, -1) # should be in the shape [1, action_size]
        state = torch.from_numpy(s_matrix).float().to(policy.device)
        action = torch.from_numpy(a_matrix).float().to(policy.device)
        values,_ , weights = policy.get_Q_prediction(state, action)
        weights = torch.softmax(weights, dim=1)
    
        return values.detach().numpy().reshape(-1), weights.detach().numpy().reshape(-1)
    
    print(video_directory)
    if not os.path.exists(video_directory):
        os.mkdir(video_directory)
    color = ["red", "blue"]
    action_split_index = []
    index = 0
    for i in model.head_config:
        index += i
        action_split_index.append(index)
    
    for seed in range(1, 2):
        
        current_directory = os.path.join(video_directory, str(seed))
        if not os.path.exists(current_directory):
            os.mkdir(current_directory)
        
        critic_values = []
        critic_values_ = []
        critic_weights = []
        action_costs = []
        rewards = []
        frames = []
        
        s = env.reset(seed=seed)[0]
        reward, done, t, truncated = 0, False, 0, False
        
        frames.append(env.render())
        count = 0
        while not (done or truncated):
            
            a = model.e_greedy_policy(s, 0, 'test')
            values, weights = get_critic_values(model, s, a)
            
            
            sp, r, done, truncated, info = env.step(a)
            s, reward, t = sp, reward + r, t + 1
            
            a_ = model.e_greedy_policy(sp, 0, 'test')
            
            values_, weights_ = get_critic_values(model, sp, a_)
            
            critic_values.append(values)
            critic_values_.append(values_)
            critic_weights.append(weights)
            rewards.append([r])

            splited_action = np.split(a, action_split_index)
            action_cost = [np.sum(i**2) for i in splited_action if not i.size==0]
            action_costs.append(action_cost)
            
            
            frames.append(env.render())
            
            count += 1      
        print("Data Collected, Starting Video Generation...")
        
        # generate video for critic values    
        # generate_a_video_for_value_array(np.array(critic_values), os.path.join(current_directory, "critic_values.mp4"))
        # generate video for predicted rewards
        # generate_a_video_for_value_array(np.array(critic_values).sum(axis=1, keepdims=True) - 0.99 * np.array(critic_values_).sum(axis=1, keepdims=True), os.path.join(current_directory, "predicted_rewards.mp4"), ["predicted rewards"])
        # generate video for head weights
        # generate_a_video_for_value_array(np.array(critic_weights),os.path.join(current_directory, "proportion_of_critics.mp4"))
        # generate video for action cost
        # generate_a_video_for_value_array(np.array(action_costs), os.path.join(current_directory, "action_costs.mp4"), moving_average=10)
        # generate_a_video_for_value_array(np.array(critic_values).sum(axis=1, keepdims=True), os.path.join(current_directory, "predicted_q_value.mp4"), ["predicted q value"])
        # print(f"episode {seed} finished with rewards as {reward}")
        # generate_a_video_for_value_array(np.array(rewards), os.path.join(current_directory, "real rewards.mp4"), ["real rewards"])
        generate_a_video_for_frame_array(frames, os.path.join(current_directory, "video.mp4"))
        
    cv2.destroyAllWindows()

def generate_videos_for_standard_TD3(env, model, video_directory):
    print(video_directory)
    if not os.path.exists(video_directory):
        os.mkdir(video_directory)
    
    for seed in range(1, 2):
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_recorder = cv2.VideoWriter(os.path.join(video_directory, str(seed)+".mp4"), fourcc, 25, (480, 480))
        s = env.reset(seed=seed)[0]
        reward, done, t, truncated = 0, False, 0, False
        
        action_costs = []
        critic_values = []
        
        count = 0
        while not (done or truncated):
            a = model.e_greedy_policy(s, 0, "test")
            a[0] = 0
            value = model.forward(
                torch.tensor(s.reshape(1, -1)).to(model.device).float(),
                torch.tensor(a.reshape(1, -1)).to(model.device).float()
            )[0]
            sp, r, done, truncated, info = env.step(a)
            s, reward, t = sp, reward + r, t + 1
            
            
            action_costs.append(np.sum(a**2))
            critic_values.append(value.detach().numpy()[0, 0])
            
            frame = env.render()
            
            video_recorder.write(frame)
            
            print(f"Frame {count} finished!", end='\r')
            count += 1
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_dir",
                        required=True,
                        type=str)
    
    args, unknowns = parser.parse_known_args()
    
    directory_list = [args.start_dir]
    file_list = []
    
    while directory_list:
        directory = directory_list.pop(0)
        for item_name in os.listdir(directory):
            f = os.path.join(directory, item_name)
            if os.path.isfile(f) and f.endswith(".pth"):
                model_name = Path(f).name
                hyperpath = os.path.join(directory, "hyperparams")
                hypernames = [file for file in os.listdir(hyperpath)]
                hypername = hypernames[0][:-12] + model_name[:-4]
                file_list.append([hyperpath, hypername,f])
            elif os.path.isdir(f):
                directory_list.append(f)
    
    for i in file_list:
        params = utils.get_hyper_parameters(i[0], i[1])
        if "env_args" in params.keys() and params["env_args"]:
            env_args = dict((a.strip(), float(b.strip()))  
                        for a, b in (element.split('-')  
                                    for element in params["env_args"].split('/')))
        else:
            env_args = dict()
            
        if params["env_name"] in ["PlainPoint", "PlainPointMax"]:
            if params["env_name"] == "PlainPoint":
                env = plainPoint(**env_args)
            elif params["env_name"] == "PlainPointMax":
                env = plainPointMAX(**env_args)
        else:
            env = gym.make(params["env_name"], render_mode="rgb_array")
        s0 = env.reset()[0]
        device = torch.device("cpu")
        print("Runing on", device)
        
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
        
        model = Constructor(params=params,
                            env = env,
                            state_size = len(s0), 
                            action_size = len(env.action_space.low), 
                            device=device)
        model.load_state_dict(torch.load(i[2], map_location=device))
        model.eval()
        
        if params["model"] == "val_decom":
            generate_videos_for_value_decomposition_TD3(env, model, i[2][:-4])
        elif params["model"] == "DDPG":
            generate_videos_for_standard_TD3(env, model, i[2][:-4])