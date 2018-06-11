import numpy as np
import matplotlib.pyplot as plt

r = .3
mu = 1

prop = .7

plt.plot(range(2), mu * np.arange(2), 'g--', marker='x')
plt.plot(range(1, 3), mu * np.arange(1, 3), 'g--', marker='x')
plt.plot([0, r], 'r', marker='x')
plt.arrow(1, r, prop * 1, prop * (2 * mu - r),
          color='red', head_width=.06)

y = [0, r, 1., 2.]
labels = ["0", "$r$", "$\\bar \mu$", "$2\ \\bar \mu$"]
plt.yticks(y, labels, rotation='vertical', fontsize=18)

x = [0, 1, 2]
labels = ["t=0", "t=1", "t=2"]
plt.xticks(x, labels, fontsize=18)
plt.show()

if __name__ == "__main__":
    pass
