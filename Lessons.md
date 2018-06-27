# Lessons Learned

Since this project requires several skills that were only briefly mentioned and not taught hands-on, it was very challenging.  Furthermore, the project was vague on what types of things I should do to make it work, so I figure I should take notes and document my discoveries to share with other students struggling with this project.

## So here they are in no particular order:
- The reward function seems to be the critical factor holding back the agent from learning properly; *do this first*.
- The starter code doesn't have any way to recognize if the target position was reached. (**Big Eureka moment right here**)
- The physics_sim doesn't have a floor, if you set `init_pos` too low and if the agent doesn't apply thrust right away, it will just fall below 0 and end the episode.
- Without specifying clear and large rewards for performing well, the agent might just decide it's better to crash early and end the episode sooner.
- Euler angles are in radians and here's what they mean:
  - `x`/`phi` corresponds to yaw
  - `y`/`theta` corresponds to roll
  - `z`/`psi` corresponds to pitch
- The angles in the pose variable are all positive, so numbers like 6.1 are actually just a small negative variance from 0  (360 degrees = 2 Pi radians)
- If implementing time-based rewards, use the time from the sim, because outside of that the time gets jumbled due to the replay buffer.
- The supplied code for a DDPG implementation is mostly plug and play (to get 'working', but tuning seems like it might be needed), but you'll probably want to add in score and noise_scale variables for training consistency with the PolicySearch_Agent.
- After an extensive grid_search on the agent's parameters (mu, theta, sigma, gamma, tau), it seems like the default ones in the sample code is optimal.
- I've heard it recommended to normalize the rewards to a range of -1 to 1, but that has the following issues:
  - Most other successful reinforcement learning implementations don't seem to do this.
  - It doesn't seem to make much sense why that would help.
  - It didn't yield much better results for me in practice.
- It's really hard to iterate quickly and try to figure out what works when it takes a long time to train, even on a high end GPU.  This is especially problematic with the DDPG approach.  Other methods might be significantly faster to run.
- This is what the params in the Ornstein–Uhlenbeck Noise Process mean:
  - mu: the mean, which controls exploration as learning progresses
  - theta: the speed of the mean reversion
  - sigma: the volatility
- The instructions suggested perhaps implementing the various types of tasks as helper functions (for example, takeoff, hover, land, etc.), but I thought it was a lot simpler and intuitive for this to be controlled directly within the `__init__` using a variable that can be passed in.
- Averaging the scores over the training period seems to be useful in seeing if the best score was just a fluke or if the agent is making serious progress.
- If programmatically trying out different values for the reward function, make sure to consider how those values affect the scores relative to the other training rounds; a higher score might just mean that crash penalties were lower.  Also, **watch out for reward hacking!**
- Adding dropout layers doesn't seem to improve the learning (nor does it seem to hinder it), but it does seem to slightly speed up the processing.
- Don't use the rewards variable in the learn function to calculate scores, as those values will be out of context from the current state of the episode.
- If you want to run training for more than a few hundred episodes, you should increase the agent's `buffer_size` or else it may wander away from the optimal policy.
