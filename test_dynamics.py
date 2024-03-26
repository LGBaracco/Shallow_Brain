import numpy as np
import math
import matplotlib.pyplot as plt

total_time = 10000
t_constant = 10  # seconds
dt = [0.01, 0.1, 1, 10]
x0 = 1


def formula(a):
    return - a / t_constant


values = np.zeros((len(dt), total_time))
analytical_values = np.zeros((len(dt), total_time))
for i in range(len(dt)):
    x = x0
    for j in range(total_time):

        x = x + dt[i] * formula(x)
        values[i, j] = x

        analytical_values[i, j] = x0 * (math.exp(-j / t_constant))

    t = np.array(range(total_time)) * dt[i]
    plt.plot(t, values[i, :], label=f'Timestep size: {dt[i]}')

plt.plot(analytical_values[0, :], label='Analytical solution', linewidth=0.5)

plt.xlim([0, 60])
plt.legend()
plt.show()

plt.clf()
