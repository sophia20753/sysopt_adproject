import pickle
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# load trained model
model = load_model('dnn.h5')

# load scaling parameters
with open('scaling_params.pkl', 'rb') as file:
    scaling_params = pickle.load(file)
mean = scaling_params['mean']
scale = scaling_params['scale']

# create duckietown environment
env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)

# reset environment
obs = env.reset()
env.render()

# define total reward
total_reward = 0

# initialise car speed and steering angle
initial_car_speed = 0.5  # Adjust as needed
initial_steering_angle = 0.0  # Adjust as needed

car_speed = initial_car_speed
steering_angle = initial_steering_angle

# main loop for simulation
while True:
    # observation data
    lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_position.dist
    angle_from_straight_in_rads = lane_position.angle_rad

    # normalise observation data 
    distance_to_road_center = (distance_to_road_center-mean[2])/scale[2]
    angle_from_straight_in_rads = (angle_from_straight_in_rads-mean[3])/scale[3]
    car_speed = (car_speed-mean[0])/scale[0]
    steering_angle = (steering_angle-mean[1])/scale[1]

    # prepare input data for the model
    input_data = np.array([
        car_speed,
        steering_angle,
        distance_to_road_center, angle_from_straight_in_rads
        ]).reshape(1, -1)
    
    # use model to predict car speed and steering angle
    prediction = model.predict(input_data)
    car_speed = prediction[0][0]
    steering_angle = prediction[0][1]

    # take a step in the environment using predicted actions
    obs, reward, done, info = env.step([car_speed, steering_angle])

    # update total reward
    total_reward += reward

    print(
        "step=%s, current reward=%.3f, total reward=%.3f"
        % (env.step_count, reward, total_reward)
    )

    env.render()

    if done:
        print('Simulation ended.')
        break

env.close()
