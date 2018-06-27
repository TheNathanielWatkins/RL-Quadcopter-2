import sys, random

def train(agent, task, num_episodes=1000):
    """
    Turned the sample code into a function so I could have some more flexibility in trying different parameters.
    """
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), Parameters = {}".format(
                    i_episode, agent.score, agent.best_score, agent.params), end="")  # [debug]
                break
        sys.stdout.flush()

    return agent

def better_grid_search(task, num_episodes=750, num_options=100, delta=0.01):
    """
    Implements a basic random-exploration gradient descent algorithm to explore optimal values.
    """
    trained_agent = train(DDPG(task), task, num_episodes)
    best_score = trained_agent.best_score
    best_agent = trained_agent

    ## Agent's params
    mu = trained_agent.exploration_mu
    theta = trained_agent.exploration_theta
    sigma = trained_agent.exploration_sigma
    gamma = trained_agent.gamma
    tau = trained_agent.tau

    for run in range(num_options):
        new_mu = round(random.uniform(mu-delta, mu+delta), 3)
        new_theta = round(random.uniform(theta-delta, theta+delta), 3)
        new_sigma = round(random.uniform(sigma-delta, sigma+delta), 3)
        new_gamma = round(random.uniform(gamma-delta, gamma+delta), 3)
        new_tau = round(random.uniform(tau-delta, tau+delta), 3)

        print("\r\n\nTrying: Mu={} || Theta={} || Sigma={} ||  Gamma={} || Tau={}".format(
            new_mu, new_theta, new_sigma, new_gamma, new_tau), end="\r\n")

        task = Task()
        new_agent = DDPG(task, mu=new_mu, theta=new_theta, sigma=new_sigma, gamma=new_gamma, tau=new_tau)
        new_agent = train(new_agent, task, num_episodes)

        if new_agent.best_score > best_score:
            best_score = new_agent.best_score
            best_agent = new_agent

            mu, theta, sigma, gamma, tau = new_mu, new_theta, new_sigma, new_gamma, new_tau

    return best_agent

# trained_agent = better_grid_search(task, num_episodes=750, num_options=100, delta=0.1)

# trained_agent = sample_agent
