from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)

# UCB : Deterministic / Requires update at every round
# TS : Probabilistic / Can accommodate delayed feedback / Better empirical evidence

# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
x = np.linspace(0, 1, 100)
ax.plot(x, beta.pdf(x, 2, 2), 'r-', lw=3, alpha=0.6, label='arm1')
ax.plot(x, beta.pdf(x, 1, 2), 'g-', lw=3, alpha=0.6, label='arm2')
ax.plot(x, beta.pdf(x, 1, 1), 'b-', lw=3, alpha=0.6, label='arm3')

banner1_rvs = beta.rvs(2, 2, size=1)
banner2_rvs = beta.rvs(1, 2, size=1)
banner3_rvs = beta.rvs(1, 1, size=1)

print("arm1:", banner1_rvs)
print("arm2:", banner2_rvs)
print("arm3:", banner3_rvs)

ax.plot(banner1_rvs, 0.0, 'x', color='red')
ax.plot(banner2_rvs, 0.0, 'x', color='green')
ax.plot(banner3_rvs, 0.0, 'x', color='blue')

ax.legend(loc='best', frameon=False)
plt.show()