from narmedbandit import BanditAgent, EpsGreedyBanditAgent, \
						 OptimisticEpsGreedyBanditAgent, UCBBanditAgent, \
						 test_eps_greedy, test_ucb
import matplotlib.pyplot as plt

if __name__ == "__main__":
	N = 10
	SHOW_PLOTS = False
	SAVE_PLOTS = True
	
	rewards, accuracy = [], []
	
	r, acc = test_eps_greedy(EpsGreedyBanditAgent, N, 0.1)
	rewards.append(r)
	accuracy.append(acc*100)
	
	r, acc = test_eps_greedy(EpsGreedyBanditAgent, N, 0.01)
	rewards.append(r)
	accuracy.append(acc*100)
	
	r, acc = test_eps_greedy(EpsGreedyBanditAgent, N, 0)
	rewards.append(r)
	accuracy.append(acc*100)
	
	
	for i in range(len(rewards)):
		plt.plot(rewards[i],linewidth=0.8)
	plt.title("Avg Reward")	
	plt.xlabel("steps")
	plt.ylabel("avg reward")
	plt.legend(["eps=0.1", "eps=0.01", "eps=0"])
	if SAVE_PLOTS:
		plt.savefig("plots/epsilon_greedy_avg_reward.png", dpi=150)
	if SHOW_PLOTS:
		plt.show()
	plt.close()
	
	for i in range(len(accuracy)):
		plt.plot(accuracy[i],linewidth=0.8)
	plt.title("Accuracy")
	plt.xlabel("steps")
	plt.ylabel("% optimal action")
	plt.legend(["eps=0.1", "eps=0.01", "eps=0"])
	if SAVE_PLOTS:
		plt.savefig("plots/epsilon_greedy_accuracy.png", dpi=150)
	if SHOW_PLOTS:
		plt.show()
	plt.close()
	
	rewards, accuracy = [], []
	
	_, acc = test_eps_greedy(EpsGreedyBanditAgent, N, 0.1)
	accuracy.append(acc)
		
	_, acc = test_eps_greedy(OptimisticEpsGreedyBanditAgent, N, 0)
	accuracy.append(acc)
		
	for i in range(len(accuracy)):
		plt.plot(accuracy[i],linewidth=0.8)
	plt.title("Accuracy")
	plt.xlabel("steps")
	plt.ylabel("% optimal action")
	plt.legend(["Q1=0,eps=0.1", "Q1=5,eps=0"])
	if SAVE_PLOTS:
		plt.savefig("plots/optimistic_epsilon_greedy_accuracy.png", dpi=150)
	if SHOW_PLOTS:
		plt.show()
	plt.close()
	
	rewards, accuracy = [], []
	
	r, _ = test_eps_greedy(EpsGreedyBanditAgent, N, 0.1)
	rewards.append(r)
		
	r, _ = test_ucb(UCBBanditAgent, N, 2)
	rewards.append(r)
		
	for i in range(len(rewards)):
		plt.plot(rewards[i],linewidth=0.8)
	plt.title("Avg Reward")
	plt.xlabel("steps")
	plt.ylabel("Avg reward")
	plt.legend(["eps=0.1", "UCB c=2"])
	if SAVE_PLOTS:
		plt.savefig("plots/ucb_avg_reward.png", dpi=150)
	if SHOW_PLOTS:
		plt.show()
	plt.close()