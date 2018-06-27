import numpy as np
from physics_sim import PhysicsSim
import random, itertools

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None,
        action_repeat=3, state_size=6, action_low=0, action_high=900,
        grid_search=False, goal="hover", min_accuracy=0.5,
        distance_factor=.59, angle_factor=1.29, v_factor=1.02, rotor_factor=.01,
        time_factor=1., elevation_factor=1.05, crash_factor=1.06, target_factor=1.04):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            action_repeat: Number of timesteps to step agent
            state_size (int): Dimension of each state
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            grid_search: Set to True to run a randomized grid_search on the reward_factor range 0.1 to 10  ## DEPRECIATED
            goal: Specifies what type of task the agent is attempting and sets standard initial values if not Otherwise specified
                Options: 'hover' (default), 'target', 'land', 'takeoff'
            min_accuracy: Sets how far the average deviation from the target point is acceptable
            distance_factor: Scalar that controls the relative reward for the distance measurement of the reward function
            angle_factor: Scalar that controls the relative reward for the angles off axis of the reward function
            v_factor: Scalar that controls the relative reward for the angular velocity of the reward function
            rotor_factor: Scalar that controls the relative reward for the relative rotor speed of the reward function
            time_factor: Scalar that controls the relative reward for the time bonus of the reward function
            elevation_factor: Scalar that controls the relative reward for the elevation measurement of the reward function
            crash_factor: Scalar that controls the relative reward for the crash detector of the reward function
            target_factor: Scalar that controls the relative reward for the goal conditions of the reward function

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
        ## Subtracts the difference between the current position & target
        reward = self.distance_factor - 0.25 * (abs(self.sim.pose[:3] - self.target_pos)).sum()

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

        ## Add max delta rotor_speeds penalty
        for rotor_a, rotor_b in itertools.combinations(rotor_speeds, 2):
            if abs(rotor_a - rotor_b) > 400:
                reward -= 10 * self.rotor_factor

        ## Set a time factor that rewards the agent for each timestep that it hasn't crashed
        ## Plus, it adds a major bonus if it makes it to the penultimate timestep.
        reward += 100 * self.time_factor
        reward = reward + (200 * self.time_factor) if self.sim.time > 4 else reward

        ## Further penalizes deviation in elevation
        if self.goal is not "land":
            reward -= abs(self.target_pos[2] - self.sim.pose[2]) * 10 * self.elevation_factor

        ## Set major reward/penalty depending on whether agent crashes early or not
        if done and np.average(abs(self.sim.pose[:3] - self.target_pos)) > self.min_accuracy * 5:
            reward -= 10000 * self.crash_factor
            # print("Crashed!! Current reward: {} || Current timestep: {}".format(reward, self.sim.time)) # Useful for visualizing behavior while training is running.

        # TODO: Add distance to target variable and use that as a type of score for evaluating model performance

        ## DEPRECIATED
        ## Normalizes the reward to a range of -1 to 1 by implementing a floor of -1
        ## This was suggested as a fix, but did not really improve the situation for me.
        # reward = -1. if reward < -1. else reward

        ## Sets a decent reward if the agent gets close to the target to help it hone in on what works
        if self.goal is not "hover" and np.average(abs(self.sim.pose[:3] - self.target_pos)) < self.min_accuracy * 4:
            reward += 100
            ## Set major reward if agent gets to the target location (within an acceptable minimum accuracy).
            ## And most importantly, it stops the simulation at that time, except hover, which can't stop early.
            if np.average(abs(self.sim.pose[:3] - self.target_pos)) < self.min_accuracy:
                ## Add an additional reward factor for reaching the target around the penultimate timestep,
                ## but not necessarily to rush there so quickly as to zoom past uncontrollably.
                reward += 10000 * (self.sim.time / (self.sim.runtime - 1)) * self.target_factor
                done = True

        if self.goal is "hover" and np.average(abs(self.sim.pose[:3] - self.target_pos)) < self.min_accuracy:
            ## Rewards the agent progressively more if it stays near the target position for a longer time.
            reward += 500 * self.target_factor

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
