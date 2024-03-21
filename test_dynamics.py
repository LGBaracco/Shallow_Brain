import numpy as np
import matplotlib.pyplot as plt

total_time = 1000
t_constant = 10  # seconds
dt = [0.01, 0.1, 1, 10, 100, 1000]
x0 = 1


def formula(a):
    return - a / t_constant


values = np.zeros((len(dt), total_time))
for i in range(len(dt)):
    x = x0
    for j in range(total_time):

        x = x + dt[i] * formula(x)
        print(x)
        values[i, j] = x

for i in range(4):
    plt.plot(values[i, :], label=f'Timestep size: {dt[i]}')

plt.legend()
plt.show()

plt.clf()
for i in range(4, 6):
    plt.plot(values[i, :], label=f'Timestep size: {dt[i]}')

plt.legend()
plt.show()