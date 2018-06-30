import numpy as np
from physics_sim import PhysicsSim
import random, itertools
from collections import deque

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None,
        action_repeat=3, state_size=6, action_low=0, action_high=900,
        goal="target", min_accuracy=2.,
        distance_factor=2.3, angle_factor=6.3, v_factor=4.5, rotor_factor=0.1,
        time_factor=0.1, elevation_factor=2.4, crash_factor=1., target_factor=1.):
        """Initialize a Task object.
        Params
        ======
            init_pose (array): initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities (array): initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities (array): initial radians/second for each of the three Euler angles
            runtime  (float): time limit for each episode
            target_pos (array): target/goal (x,y,z) position for the agent
            action_repeat (int): Number of timesteps to step agent
            state_size (int): Dimension of each state
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            goal (string): Specifies what type of task the agent is attempting and sets standard initial values if not Otherwise specified
                Options: 'hover' (default), 'target', 'land', 'takeoff'
            min_accuracy (float): Sets how far the average deviation from the target point is acceptable
            distance_factor (float): Scalar that controls the relative reward for the distance measurement of the reward function
            angle_factor (float): Scalar that controls the relative reward for the angles off axis of the reward function
            v_factor (float): Scalar that controls the relative reward for the angular velocity of the reward function
            rotor_factor (float): Scalar that controls the relative reward for the relative rotor speed of the reward function
            time_factor (float): Scalar that controls the relative reward for the time bonus of the reward function
            elevation_factor (float): Scalar that controls the relative reward for the elevation measurement of the reward function
            crash_factor (float): Scalar that controls the relative reward for the crash detector of the reward function
            target_factor (float): Scalar that controls the relative reward for the goal conditions of the reward function

        """
        ## Set target and initial conditions based on the goal
        ## But doesn't override if other conditions were specified
        if goal is "target":
            self.target_pos = np.array([10., 0., 10.]) if target_pos is None else target_pos

        elif goal is "hover":
            self.target_pos = np.array([0., 0., 10.]) if target_pos is None else target_pos

        elif goal is "land":
            self.target_pos = np.array([0., 0., 0.01]) if target_pos is None else target_pos

        elif goal is "takeoff":
            self.target_pos = np.array([0., 0., 10.]) if target_pos is None else target_pos
            init_pose = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]) if init_pos is None else init_pos
            init_velocities = np.array([0.0, 0.0, 0.0]) if init_velocities is None else init_velocities
            init_angle_velocities = np.array([0.0, 0.0, 0.0]) if init_angle_velocities is None else init_angle_velocities

        else:
            print("It seems an improper goal was specified.  Please input one of the following options:\
                * 'hover'\n* 'target'\n* 'land'\n* 'takeoff'")

        self.goal = goal

        ## Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = action_repeat

        self.state_size = self.action_repeat * state_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_size = 4 # Dimension of each action (one for each rotor)

        self.min_accuracy = min_accuracy
        # self.reward_factor = reward_factor  # Default value is 1 ## DEPRECIATED

        self.distance_factor = distance_factor
        self.angle_factor = angle_factor
        self.v_factor = v_factor
        self.rotor_factor = rotor_factor
        self.time_factor = time_factor
        self.elevation_factor = elevation_factor
        self.crash_factor = crash_factor
        self.target_factor = target_factor

        ## Numerical representation of the distance to the target
        self.target_proximity = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        self.total_proximity = deque(maxlen=500)
        self.average_proximity = 0.
        self.best_proximity = np.inf
        self.previous_proximity = self.target_proximity

        self.params = {
        "Distance" : self.distance_factor, "Angle" : self.angle_factor, "Angular V" : self.v_factor, "Rotor" : self.rotor_factor,
        "Time" : self.time_factor, "Elevation" : self.elevation_factor, "Crash" : self.crash_factor, "Target" : self.target_factor}

    def get_reward(self, previous_reward, done, rotor_speeds):
        """
        Returns a reward based on a number of positive and negative factors guiding the agent on how to perform.
        These values seemed like appropriate numbers to encourage good behavior,
        but I added scaler variables to try various relations between the weights.
        Overall, the goal was for the agent to have a negative score if it crashes, a positive score if it doesn't,
        and it should have a very positive score if it achieves it's goal.
        """
        ## Updates the distance to target and calculates total, aveerage & best if done
        self.target_proximity = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        if done:
            self.total_proximity.append(self.target_proximity)
            self.average_proximity = np.mean(self.total_proximity)
            if self.target_proximity < self.best_proximity:
                self.best_proximity = self.target_proximity

        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()  # Default reward function

        ## Subtracts the difference between the current position & target
        reward = (1 - self.target_proximity) * self.distance_factor

        ## Penalizes when moving further from the target than before, within a margin of error
        if self.previous_proximity < (self.target_proximity * 1.2):
            reward -= 50 * self.distance_factor
        self.previous_proximity = self.target_proximity

        ## Adds a reward amount if the yaw, roll or pitch are less than about 28 degrees,
        ## Otherwise, it's a negative reward
        for angle in self.sim.pose[3:]:
            if angle >= 6.3:
                print("Something's wrong with the angles.")
            if angle >= 3.1415:
                angle = 6.283 - angle

            reward += (.5 - angle) * self.angle_factor

        ## Adds a reward amount if the yaw, roll or pitch velocities are less than about 28 degrees per second,
        ## Otherwise, it's a negative reward
        for v in self.sim.angular_v:
            reward += (.5 - abs(v)) * self.v_factor

        ## Add a penalty if the rotors differ in speed too drastically
        for rotor_a, rotor_b in itertools.combinations(rotor_speeds, 2):
            if abs(rotor_a - rotor_b) > 500:
                reward -= abs(rotor_a - rotor_b) * self.rotor_factor

        ## Set a time factor that rewards the agent for each timestep that it hasn't crashed
        ## Plus, it adds a major bonus if it makes it to the penultimate timestep.
        reward += 100 * self.time_factor
        reward = reward + (100 * self.time_factor) if self.sim.time > 4 else reward

        ## Further penalizes deviation in elevation
        if self.goal is not "land":
            reward -= abs(self.target_pos[2] - self.sim.pose[2]) * 10 * self.elevation_factor

        ## Set major reward/penalty depending on whether agent crashes early or not
        if done and np.average(abs(self.sim.pose[:3] - self.target_pos)) > self.min_accuracy * 2:
            reward -= 10000 * self.crash_factor
            # print("Crashed!! Current proximity: {} || Current timestep: {}".format(self.target_proximity, self.sim.time)) # Useful for visualizing behavior while training is running.

        ## Sets a decent reward if the agent gets close to the target to help it hone in on what works
        if self.target_proximity < self.min_accuracy * 4:
            reward += 500 * self.target_factor
            ## Set major reward if agent gets to the target location (within an acceptable minimum accuracy).
            ## And most importantly, it stops the simulation at that time, except hover, which can't stop early.
            if self.target_proximity < self.min_accuracy:
                ## Add an additional reward factor for reaching the target around the penultimate timestep,
                ## but not necessarily to rush there so quickly as to zoom past uncontrollably.
                reward += 10000 * (self.sim.time / (self.sim.runtime - 1)) * self.target_factor
                done = True if self.goal is not "hover" else False
                print("Success!  (•̀ᴗ•́)و ̑̑  ")

        ## DEPRECIATED
        ## Normalizes the reward to a range of -1 to 1 by implementing a floor of -1
        ## This was suggested as a fix, but did not really improve the situation for me.
        # reward = -1. if reward < -1. else reward

        return reward + previous_reward, done

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward, done = self.get_reward(reward, done, rotor_speeds)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
