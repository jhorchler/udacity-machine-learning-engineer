import numpy as np
from physics_sim import PhysicsSim
from typing import Tuple


class ExampleTask:
    """Task (environment) that defines the goal and provides feedback to the
    agent. """

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.

        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions
                       and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z)
                             dimensions
            init_angle_velocities: initial radians/second for each of the three
                                   Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        done = False
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state


class Task:
    """Implements a Quadcopter Task"""

    def __init__(self, init_pose: np.array = None,
                 init_velocities: np.array = None, action_repeat: int = 3,
                 init_angle_velocities: np.array = None, runtime: float = 5.,
                 target_pos: np.array = None) -> None:
        """Creates a new Task"""
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = action_repeat
        self.state_size = self.action_repeat * len(init_pose)
        self.action_low = 0.1
        self.action_high = 900
        self.action_size = 4
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10., 0., 0., 0.])
        # TODO: handle case where init_pose eq None
        self.initial_distance = np.linalg.norm(
            self.target_pos[:3] - init_pose[:3])

    def get_reward(self) -> float:
        """Uses current pose of sim to return reward."""
        # lower_bounds: array([-150., -150.,    0.])
        # upper_bounds: array([ 150.,  150., 300.])

        # Udacity Review - take original reward function
        reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos[:3])).sum()
        reward = np.tanh(reward)

        # start with nothing
        # reward = 0

        # ADD reward that is higher as closer the quadcopter is the at the
        # target position. Idea is that the reward gets more if the quadcopter
        # comes closer to the target position.
        # current_distance = np.linalg.norm(self.target_pos[:3] -
        #                                   self.sim.pose[:3])
        # if current_distance < self.initial_distance:
        #     reward += 100 * (self.initial_distance - current_distance)
        # else:
        #     reward -= current_distance

        # punishment if quadcopter hits the wall
        # for idx in range(3):
        #     if (int(self.sim.pose[idx]) == int(self.sim.lower_bounds[idx]) or
        #             int(self.sim.pose[idx]) == int(self.sim.upper_bounds[idx])):
        #         reward -= 10

        # raise reward if target is touched
        # for idx in range(3):
        #     if int(self.sim.pose[idx]) == int(self.target_pos[idx]):
        #         reward += 1

        # raise reward for positive velocity
        # for idx in range(3):
        #     if self.sim.v[idx] > 0:
        #         reward += 1

        # maximize return if all positions are identical (more or less,
        # as floats can not be compared that easy)
        if (int(self.target_pos[0]) == int(self.sim.pose[0]) and
                int(self.target_pos[1]) == int(self.sim.pose[1]) and
                int(self.target_pos[2]) == int(self.sim.pose[2])):
            reward += 100

        return reward

    def step(self,
             rotor_speeds: list) -> Tuple[np.ndarray, float, bool]:
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        pose_all = []
        done = False
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
