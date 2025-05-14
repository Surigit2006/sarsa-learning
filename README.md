# SARSA Learning Algorithm


## AIM
To implement SARSA Learning Algorithm.

## PROBLEM STATEMENT

The problem might involve teaching an agent to interact optimally with an environment (e.g., gym-walk), where the agent must learn to choose actions that maximize cumulative rewards using RL algorithms like SARSA and Value Iteration.


## SARSA LEARNING ALGORITHM

1.Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.

2.For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.

3.Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.

4.Update the Q-value for the state-action pair (s,a) using the SARSA update rule.

5.Update the current state to s′ and the current action to a′.

6.Repeat steps 3-5 until the episode reaches a terminal state.

7.After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.

8.Return the Q-table and the learned policy after completing all episodes.


## SARSA LEARNING FUNCTION
### Name:M.K.Suriya prakash
### Register Number:212224110053

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    alphas = decay_schedule(
        init_alpha, min_alpha,
        alpha_decay_ratio, n_episodes)

    epsilons = decay_schedule(
        init_epsilon, min_epsilon,
        epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * \
                        Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + \
                               alphas[e] * td_error

            state, action = next_state, next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(
        np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
## optimal policy, optimal value function , success rate for the optimal policy.
![Screenshot 2025-05-14 114411](https://github.com/user-attachments/assets/8088cadb-fc33-450b-b820-9b3bc499db11)


![Screenshot 2025-05-14 114440](https://github.com/user-attachments/assets/7e4aaa04-7366-4b51-ada0-09ea2f0849ff)

![Screenshot 2025-05-14 114448](https://github.com/user-attachments/assets/501ac66a-a90d-4e38-a1d0-4111e60f008f)


## plot comparing the state value functions of Monte Carlo method and SARSA learning.

![Screenshot 2025-05-14 114500](https://github.com/user-attachments/assets/8f1843b6-f895-45fd-a287-2fb863ab5fa4)

![Screenshot 2025-05-14 114511](https://github.com/user-attachments/assets/9b2c1d41-eafd-4511-8d51-142a0c02c2ba)


## RESULT:
Thus to implement SARSA learning algorithm is executed successfully.
