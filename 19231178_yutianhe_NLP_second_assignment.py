import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义高斯分布的参数
mean1, std1 = 164, 3
mean2, std2 = 176, 5

np.random.seed(0)
data1 = np.random.normal(mean1, std1, 500)
np.random.seed(1)
data2 = np.random.normal(mean2, std2, 1500)
data = np.concatenate((data1, data2), axis=0)

# 初始化参数
K = 2  # 混合高斯分布的个数
pi = np.ones(K) / K  # 混合系数
mu = np.random.choice(data, size=K)  # 均值
sigma = np.ones(K)  # 方差

pi_lst1 = [pi[0]]
mu_lst1 = [mu[0]]
sigma_lst1 = [sigma[0]]
pi_lst2 = [pi[1]]
mu_lst2 = [mu[1]]
sigma_lst2 = [sigma[1]]

# EM算法迭代更新参数
for i in range(300):
    # E步：计算后验概率
    post_prob = np.zeros((K, len(data)))
    for k in range(K):
        post_prob[k] = pi[k] * norm.pdf(data, loc=mu[k], scale=sigma[k])
    post_prob /= post_prob.sum(axis=0)

    # M步：更新参数
    for k in range(K):
        pi[k] = post_prob[k].mean()
        mu[k] = (post_prob[k] * data).sum() / post_prob[k].sum()
        sigma[k] = np.sqrt((post_prob[k] * (data - mu[k]) ** 2).sum() / post_prob[k].sum())
    pi_lst1.append(pi[0])
    mu_lst1.append(mu[0])
    sigma_lst1.append(sigma[0])
    pi_lst2.append(pi[1])
    mu_lst2.append(mu[1])
    sigma_lst2.append(sigma[1])


num_point = 100
boundary = 0
x = np.linspace(start=data.min(), stop=data.max(), num=num_point)
y_total = np.zeros_like(x)

for k in range(K):
    y_total += pi[k] * norm.pdf(x, loc=mu[k], scale=sigma[k])

y_single_lst = [pi[k] * norm.pdf(x, loc=mu[k], scale=sigma[k]) for k in range(K)]

flag = y_single_lst[0][0] > y_single_lst[1][0]
for i in range(1, num_point):
    if flag != (y_single_lst[0][i] > y_single_lst[1][i]):
        boundary = x[i]
        break


# 绘制结果
plt.figure(1)
plt.plot([i for i in range(0, len(pi_lst1))], pi_lst1, label='ingredient1')
plt.plot([i for i in range(0, len(pi_lst2))], pi_lst2, label='ingredient2')
plt.xlabel('iter')
plt.ylabel('value')
plt.title('Changes in parameter pi')
plt.legend()
#plt.show()
plt.savefig('pi_plot.png')

plt.figure(2)
plt.plot([i for i in range(0, len(mu_lst1))], mu_lst1, label='ingredient1')
plt.plot([i for i in range(0, len(mu_lst2))], mu_lst2, label='ingredient2')
plt.xlabel('iter')
plt.ylabel('value')
plt.title('Changes in parameter mu')
plt.legend()
#plt.show()
plt.savefig('mu_plot.png')

plt.figure(3)
plt.plot([i for i in range(0, len(sigma_lst1))], sigma_lst1, label='ingredient1')
plt.plot([i for i in range(0, len(sigma_lst2))], sigma_lst2, label='ingredient2')
plt.xlabel('iter')
plt.ylabel('value')
plt.title('Changes in parameter sigma')
plt.legend()
#plt.show()
plt.savefig('sigma_plot.png')

plt.figure(4)
plt.hist(data, bins=20, alpha=0.5, density=True, label='data')
plt.plot(x, y_total, 'r-', linewidth=2, label='mixture gaussian')
plt.legend()
plt.xlabel('Height (cm)')
plt.ylabel('Probability')
plt.title('Gaussian Mixture Model')
#plt.show()
plt.savefig('mixture_plot.png')

plt.figure(5)
for k in range(K):
    plt.plot(x, y_single_lst[k], label='ingredient'+str(k+1))
plt.axvline(x=boundary, color='red', label='boundary: height='+str(boundary)+'cm')
plt.legend()
plt.xlabel('Height (cm)')
plt.ylabel('Probability')
#plt.show()
plt.savefig('ingredient_plot.png')


np.random.seed(2)
data3 = np.random.normal(mean1, std1, 500)
np.random.seed(3)
data4 = np.random.normal(mean2, std2, 1500)
print((np.count_nonzero(data3 < boundary) +
       np.count_nonzero(data4 >= boundary))/2000)