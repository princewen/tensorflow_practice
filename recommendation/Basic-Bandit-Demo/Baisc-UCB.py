import numpy as np

T = 1000  # T轮试验
N = 10  # N个老虎机

true_rewards = np.random.uniform(low=0, high=1, size=N)  # 每个老虎机真实的吐钱概率
estimated_rewards = np.zeros(N)  # 每个老虎机吐钱的观测概率，初始都为0
chosen_count = np.zeros(N)  # 每个老虎机当前已经探索的次数，初始都为0
total_reward = 0


# 计算delta
def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])

# 计算每个老虎机的p+delta，同时做出选择
def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, item) for item in range(N)]
    item = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward


for t in range(1, T):  # 依次进行T次试验
    # 选择一个老虎机，并得到是否吐钱的结果
    item, reward = UCB(t, N)
    total_reward += reward  # 一共有多少客人接受了推荐

    # 更新每个老虎机的吐钱概率
    estimated_rewards[item] = ((t - 1) * estimated_rewards[item] + reward) / t
    chosen_count[item] += 1
