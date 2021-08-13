# N-Armed-Bandit

Replicas of the images about N-Armed-Bandit in the book "Reinforcement Learning: An Introduction".

Agents implemented in `narmedbandit.py`:

- `BanditAgent` (boilerplate class)

- `EpsGreedyBanditAgent`, which uses epsilon greedy to balance exploration and exploitation

  ![Eps-Greedy](/Users/simone/Documents/programming/narmedbandit/plots/epsilon_greedy_avg_reward.png)

  ![Eps-Greedy](/Users/simone/Documents/programming/narmedbandit/plots/epsilon_greedy_accuracy.png)

- `OptimisticEpsGreedyBanditAgent`, an epsilon greedy agent with higher initial action values

  ![Optimistic-Eps-Greedy](/Users/simone/Documents/programming/narmedbandit/plots/optimistic_epsilon_greedy_accuracy.png)

- `UCBBanditAgent`, which uses UCB scores to balance exploration and exploitation

  ![UCB-Agent](/Users/simone/Documents/programming/narmedbandit/plots/ucb_avg_reward.png)