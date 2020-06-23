import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys

labels = ['DeepSolar-1.0', 'Default', 'RobustMPC']
# SCHEMES = ['sim_rl', 'sim_ppo']
# labels = ['Buffer-based', 'RobustMPC']
LW = 2.5

def U(t):
    if t <= 349: return 0.12
    if t <= 528: return 0.25
    if t <= 872: return 0.5
    if t <= 1042: return 0.25
    return 0.12

def main():
    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----
    arr = []
    f = open('1.csv', 'r')
    for line in f:
        arr.append(float(line))
    f.close()

    arr0 = []
    for p in range(1400):
        arr0.append(U(p))
    
    ARR = [arr, arr0]
    plt.rcParams['axes.labelsize'] = 16
    font = {'size': 14}
    matplotlib.rc('font', **font)
    #matplotlib.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.97, top=0.97)

    lines = ['-', '--', '-.', ':', '--']
    #colors = ['red', 'blue', 'orange', 'green', 'black']

    def rgb_to_hex(rr, gg, bb):
        rgb = (rr, gg, bb)
        return '#%02x%02x%02x' % rgb

    colors = [rgb_to_hex(237, 65, 29), rgb_to_hex(102, 49, 160), rgb_to_hex(
        255, 192, 0), rgb_to_hex(29, 29, 29), rgb_to_hex(0, 212, 97)]
        
    for (arr, color, line, label) in zip(ARR, colors, lines, labels):
        ax.plot(arr, line, color=color, lw=LW, label=label)

    ax.legend(framealpha=1,
              frameon=False, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.xlim(0.25, 2.5)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.ylim(0., 0.8)
    plt.ylabel('U')
    # plt.grid(True, axis='y')
    plt.xlabel('Time')
    savefig('0.png')


if __name__ == '__main__':
    main()
