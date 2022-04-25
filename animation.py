"""
Codes for animation

Using:
matplotlib: 3.4.1
"""
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import *
from collections import deque


def animate(i):
    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(states[i, 0, 0])
    history_y.appendleft(states[i, 0, 1])
    trace1.set_data(history_x, history_y)
    trace2.set_data(states[i, 0, 3], states[i, 0, 4])
    time_text.set_text(time_template % (i * dt))
    return trace1, time_text


if __name__ == '__main__':
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    states = np.load(result_dir + '/path_global.npy')
    # states = np.load(result_dir + '/path_test.npy')

    scenario = '2Ships_Cross'
    # scenario = '3Ships_Cross&Headon'
    env = get_data(scenario)

    dt = 0.05
    t_step = len(states)
    t_stop = t_step * dt
    history_len = 100

    x = states[:, 0, 0]
    y = states[:, 0, 1]

    for i in range(1, env.ships_num):
        x = np.r_[x, states[:, 0, 3 * i]]
        y = np.r_[x, states[:, 0, 3 * i + 1]]
    x_min = np.min(x) - 100
    x_max = np.max(x) + 100
    y_min = np.min(y) - 100
    y_max = np.max(y) + 100

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.set_aspect('equal')

    trace1, = ax.plot([], [], '.-', lw=1, ms=2)
    trace2, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

    ani = animation.FuncAnimation(
        fig, animate, len(states), interval=dt * 1000, blit=True)
    plt.show()
