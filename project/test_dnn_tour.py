# import relevant libraries
import pickle
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pandas as pd
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import csv
import random
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from project_functions import save_data_to_csv

def process_input(env, mean, scale):
    lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_position.dist
    angle_from_straight_in_rads = lane_position.angle_rad

    # normalise observation data 
    distance_to_road_center = (distance_to_road_center-mean[1])/scale[1]
    angle_from_straight_in_rads = (angle_from_straight_in_rads-mean[2])/scale[2]

    input_data = np.array([[distance_to_road_center, angle_from_straight_in_rads]])

    return input_data

def main(model_name, map_name, data_type):
    # create environment with training map
    env = DuckietownEnv(map_name=map_name, domain_rand=False, draw_bbox=False, user_tile_start=(1, 1))
    env.render()
    env.max_steps = 5000 # increase max step count to ensure robot finishes traversing map

    # load model
    model = load_model(f"models/{model_name}.h5")

    # load scaling parameters
    with open(f'{data_type}_scaling_params.pkl', 'rb') as file:
        scaling_params = pickle.load(file)
    mean = scaling_params['mean']
    scale = scaling_params['scale']

    # create folder for test data
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # initialise test data for analysis
    test_model_data = {
        "cur_pos_x": [],
        "cur_pos_z": [],
        "distance_to_road_center": [],
        "angle_from_straight_in_rads": [],
        "steering_angle": [],
        "reward": [],
    }

    # make initial step with default speed and angle
    car_speed = 0.1
    steering_angle = 0

    _, reward, done, _ = env.step([car_speed, steering_angle])

    total_reward = 0

    while True:
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        # update total reward
        total_reward += reward

        input_data = process_input(env, mean, scale)

        # make prediction
        steering_angle = model.predict(input_data)

        # reverse normalisation
        steering_angle = steering_angle*scale[0] + mean[0]

        _, reward, done, _ = env.step([car_speed, steering_angle])

        # update display
        env.render()

        # store timestep data
        test_model_data["cur_pos_x"].append(env.cur_pos[0])
        test_model_data["cur_pos_z"].append(env.cur_pos[2])
        test_model_data["distance_to_road_center"].append(distance_to_road_center)
        test_model_data["angle_from_straight_in_rads"].append(angle_from_straight_in_rads)
        test_model_data["steering_angle"].append(steering_angle)
        test_model_data["reward"].append(reward)

        if done:
            if reward < 0:
                print(f"Test crashed, run again")
                env.reset()
                # reset data collection 
                # initialise test data for analysis
                test_model_data = {
                    "cur_pos_x": [],
                    "cur_pos_z": [],
                    "distance_to_road_center": [],
                    "angle_from_straight_in_rads": [],
                    "steering_angle": [],
                    "reward": [],
                }
                continue
            else:
                # successful run so save data to csv
                print(f"Test reached goal")
                save_data_to_csv("test_data", f"test_{model_name}_data.csv", test_model_data)
                break

    

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Run Duckietown simulation and collect data')
    parser.add_argument('--model_name', type=str, help='Name of .h5 file')
    parser.add_argument('--map_name', type=str, help='Name of map to simulate in')
    parser.add_argument('--data_type', type=str, help='PD/MPC data for scaling parameters')
    args = parser.parse_args()

    main(args.model_name, args.map_name, args.data_type)