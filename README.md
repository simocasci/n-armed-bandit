- # N-Armed-Bandit

  Replicas of the images about N-Armed-Bandit in the book "Reinforcement Learning: An Introduction".
  
  Agents implemented in `narmedbandit.py`:
  
  - `BanditAgent` (boilerplate class)
  
  - `EpsGreedyBanditAgent`, which uses epsilon greedy to balance exploration and exploitation
  
    ![epsilon_greedy_avg_reward](https://github.com/simocasci/n-armed-bandit/blob/main/plots/epsilon_greedy_avg_reward.png)
  
    ![epsilon_greedy_accuracy](https://github.com/simocasci/n-armed-bandit/blob/main/plots/optimistic_epsilon_greedy_accuracy.png)
  
  - `OptimisticEpsGreedyBanditAgent`, an epsilon greedy agent with higher initial action values
  
    ![optimistic_epsilon_greedy_accuracy](https://github.com/simocasci/n-armed-bandit/blob/main/plots/optimistic_epsilon_greedy_accuracy.png)
  
  - `UCBBanditAgent`, which uses UCB scores to balance exploration and exploitation
  
    ![ucb_avg_reward](https://github.com/simocasci/n-armed-bandit/blob/main/plots/ucb_avg_reward.png)