"""
Codes for animation

Using:
matplotlib: 3.4.1
ffmpeg: 2.7.0
"""
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import *
from collections import deque


def animate(i):
    time_text.set_text(time_template % (i * dt))
    for n in range(env.ships_num):
        if i > 0:
            headings[n] = ax.patches.remove(headings[n])
        past_trajectory[n].set_xdata(states[:i, 0, 3 * n])
        past_trajectory[n].set_ydata(states[:i, 0, 3 * n + 1])
        if i > 0:
            dx = states[i, 0, 3 * n] - states[i-1, 0, 3 * n]
            dy = states[i, 0, 3 * n + 1] - states[i-1, 0, 3 * n + 1]
            headings[n] = ax.arrow(states[i, 0, 3 * n], states[i, 0, 3 * n + 1], dx, dy,
                                   head_width=800, head_length=800, fc=colors[n], ec=colors[n])
    # return time_text, ship_markers, past_trajectory


if __name__ == '__main__':
    result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    states = np.load(result_dir + '/path_global.npy')
    # states = np.load(result_dir + '/path_test.npy')

    scenario = '2Ships_Cross'
    # scenario = '3Ships_Cross&Headon'
    env = get_data(scenario)

    dt = 1
    t_step = len(states)

    x = states[:, 0, 0]
    y = states[:, 0, 1]

    for i in range(1, env.ships_num):
        x = np.r_[x, states[:, 0, 3 * i]]
        y = np.r_[x, states[:, 0, 3 * i + 1]]
    x_min = np.min(x) - 500
    x_max = np.max(x) + 500
    y_min = np.min(y) - 500
    y_max = np.max(y) + 500

    past_trajectory = []
    headings = []
    colors = ['b', 'r', 'g', 'y', 'c', 'm']
    # colors = ['blue', 'purple', 'darkolivegreen', 'teal', 'darkorange', 'saddlebrown']

    fig = plt.figure()
    ax = fig.add_subplot(autoscale_on=False, xlim=(x_min, x_max), ylim=(y_min, y_max))
    ax.set_aspect('equal')

    for i in range(env.ships_num):
        past_trajectory.append(ax.plot([], [], c=colors[i], alpha=0.8)[0])
        headings.append(ax.arrow([], [], [], []))
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    frequency = 1  # when this value is set lower than 1, the
    ani = animation.FuncAnimation(
        fig, animate, len(states), interval=frequency, blit=False)
    plt.show()

    # ani.save("result.gif", writer='pillow')
    ani.save(result_dir + '/animation.mp4', writer='ffmpeg', fps=1000 / 20)