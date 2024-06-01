#!/usr/bin/env python3

"""
Tansu Alpcan - modified from basic_control.py script in github
v1 2022-3-14

Contents:
* Creates a duckietown environment
* Shows how ground truth data can be sampled from the simulator
* Gives an example of PD control applied to sim data
"""

# import relevant libraries
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv


def main():
    # create environment with a map
    env = DuckietownEnv(map_name="udem1", domain_rand=False, draw_bbox=False)
    env.reset()
    env.render()

    # define total reward
    total_reward = 0

    # main loop
    while True:
        # getting our lane position
        # in practice these should be provided by the camera/vision system!
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        # let's try PD control!
        k_p = 10
        k_d = 1

        car_speed = 0.1  # fix speed, forward direction

        # decide on steering using PD control
        steering_angle = (
            k_p * distance_to_road_center + k_d * angle_from_straight_in_rads
        )
        # You should use your control skills here!
        # Try adaptive PID or model-predictive control, for example!
        # And/or you can use your reinforcement learning skills!

        '''
        Information about how Actions are defined:

        The simulator uses continuous actions by default.
        Actions passed to the step() function should be numpy arrays containining
        two numbers between -1 and 1. These two numbers correspond to forward
        velocity, and a steering angle, respectively. A positive velocity makes the
        robot go forward, and a positive steering angle makes the robot turn left.
        There is also a Gym wrapper class named DiscreteWrapper which allows you to
        use discrete actions (turn left, move forward, turn right) instead of
        continuous actions if you prefer.

        Information about Rewards:
        reward_range = (-1000, 1000)

        reward is computed in simulator.py by the function
        compute_reward(self, pos, angle, speed)
        '''
        obs, reward, done, info = env.step([car_speed, steering_angle])

        # update total reward
        total_reward += reward

        print(
            "step=%s, current reward=%.3f, total reward=%.3f"
            % (env.step_count, reward, total_reward)
        )

        # let the simulator update its display
        env.render()

        if done:
            if reward < 0:
                print("*** CRASHED ***")
            print("Exiting")
            print(info)
            break


if __name__  == '__main__':
    main()
