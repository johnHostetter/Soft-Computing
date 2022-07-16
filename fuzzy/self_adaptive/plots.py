#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:48:51 2022

@author: john
"""

import os
import gym
import torch
import random
import numpy as np
import pandas as pd

# load package
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def plt_term(min_x, max_x, params, func, lbl, style, fill_between):
    X = np.linspace(min_x, max_x, num=2000)
    Y = [func(x, params) for x in X]
    if fill_between:
        plt.fill_between(X, [float(y) for y in Y], label=lbl, hatch='///', edgecolor='#0bafa9', facecolor='None')
    else:
        plt.plot(X, Y, label=lbl, color='k', ls=style, linewidth=2.5)
    # plt.plot(X, Y, label=lbl, ls=style)


def gaussian(x, center, sigma, last=False):
    if last and x > center:
        return 1.0
    else:
        return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


def gaussian_params(x, params):
    return gaussian(x, params['center'], params['sigma'], False)


# clip results figure

from collections import OrderedDict

antecedents_df = pd.read_csv('0_antecedents.csv')
print(antecedents_df.loc[antecedents_df['input_variable'] == 0].to_dict())

# linestyles = OrderedDict(
#     [
#      ('solid',               (0, ())),
#      # ('loosely dotted',      (0, (1, 10))),
#      ('dotted',              (0, (1, 5))),
#      # ('densely dotted',      (0, (1, 1))),

#      # ('loosely dashed',      (0, (5, 10))),
#      ('dashed',              (0, (5, 5))),
#      # ('densely dashed',      (0, (5, 1))),

#      # ('loosely dashdotted',  (0, (3, 10, 1, 10))),
#      ('dashdotted',          (0, (3, 5, 1, 5))),
#      ('densely dashdotted',  (0, (3, 1, 1, 1))),

#      # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
#      ]
#     )

# linestyles = OrderedDict(
#     [
#       ('dotted',              (0, (1, 5))),
#      # # ('loosely dotted',      (0, (1, 10))),
#      ('densely dotted',      (0, (1, 1))),

#      # ('loosely dashed',      (0, (5, 10))),
#      # ('dashed',              (0, (5, 5))),
#      ('densely dashed',      (0, (5, 1))),
#      ('solid',               (0, ())),

#      # ('loosely dashdotted',  (0, (3, 10, 1, 10))),
#      ('densely dashdotted',  (0, (3, 1, 1, 1))),

#      # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ]
#     )

# feb 7, 2022
linestyles = OrderedDict(
    [
        # ('loosely dotted',      (0, (1, 10))),
        ('dotted', (0, (1, 5))),
        ('loosely dashed', (0, (5, 10))),

        ('densely dotted', (0, (1, 1))),

        # ('loosely dashed',      (0, (5, 10))),
        # ('dashed',              (0, (5, 5))),
        # ('densely dashed',      (0, (5, 1))),
        ('solid', (0, ())),

        # ('loosely dashdotted',  (0, (3, 10, 1, 10))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ]
)

import gym
import matplotlib.pyplot as plt

from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box


class CartPole:
    def __init__(self, config=None):
        self.env = gym.make("CartPole-v1")
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def set_state(self, state):
        self.env.state = self.env.unwrapped.state = deepcopy(state)
        return self.env.state

    def get_state(self):
        return deepcopy(self.env)

    def render(self, mode):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


cartpole = CartPole()
cartpole.reset()
background_img = cartpole.render(mode='rgb_array')
# plt.imshow(background_img, alpha=0.95)
print(cartpole.env.unwrapped.state)
import torch

cartpole.set_state(np.array([-1.3607662, -0.013676725, -0.13278057, -1.7408595]).astype(dtype='float32'))
print(cartpole.env.state)
overlay_img = cartpole.render(mode='rgb_array')
# plt.imshow(overlay_img, alpha=0.95)
try:
    from PIL import Image
except ImportError:
    import Image
background = Image.fromarray(background_img)
overlay = Image.fromarray(overlay_img)
# background = Image.open("bg.png")
# overlay = Image.open("ol.jpg")
new_img = Image.blend(overlay, background, 0.25)
# plt.imshow(new_img, alpha=1., extent=[-100, 500, 400, 0])
fig, ax = plt.subplots()
plt.imshow(new_img, alpha=1.)

plt.text(275, 140, '(1)', fontsize=16)
plt.text(100, 140, '(2)', fontsize=16)

# # draw arc
# from numpy import sin, cos, pi, linspace
#
# r = 100
# precision = 100
# arc_angles = linspace(pi, 7 * pi / 4, precision)
# arc_xs = r * cos(arc_angles) + 150
# arc_ys = r * sin(arc_angles) + 300
# plt.plot(arc_xs, arc_ys, color='red', lw=3)
# # plt.gca().annotate('Arc', xy=(1.5, 100), xycoords='data', fontsize=10, rotation=120)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# plt.axes().set_xlim(-100, 500)
# plt.axes().set_ylim(-0.9,0.7)
# plt.axes().set_aspect(1)

style = "Simple, tail_width=0.6, head_width=5, head_length=8"
kw = dict(arrowstyle=style, color="#df3d3f", linewidth=3)

# a1 = patches.FancyArrowPatch((-0.4, -0.6), (0, 0.6), **kw)
a2 = patches.FancyArrowPatch((300, 300), (160, 300), **kw)
a3 = patches.FancyArrowPatch((100, 200), (50, 300),
                             connectionstyle="arc3,rad=.1327", **kw)

style = "Simple, tail_width=0.6, head_width=5, head_length=8"
kw = dict(arrowstyle=style, color="#0bafa9", linewidth=3)

move_left = patches.FancyArrowPatch((85, 350), (15, 350), **kw)
plt.text(90, 360, '$Rule_{21}$', fontsize=16)
move_right = patches.FancyArrowPatch((180, 350), (250, 350), **kw)
plt.text(-70, 360, '15.90', fontsize=16)
plt.text(260, 360, '15.72', fontsize=16)

for a in [a2, a3, move_left, move_right]:
    plt.gca().add_patch(a)
plt.axis('off')
fig.savefig('cart_pole_rule_21.svg', format='svg', dpi=1200)
plt.show()

cartpole.close()

styles = [item[1] for item in list(linestyles.items())]

# line_styles = ['--', '-.', '-', ':', '^', '|']
mins = [-2, -5, -.2, -4.5]
maxes = [2, 5, .2, 5]
plt_lbls = ['a', 'b', 'c', 'd']
selected_terms_for_each_input = [1, 0, 1, 1]
offsets = [0.1, 0., 0., 0.15]
ticks = [1.0, 1.0, 0.1, 1.0]
state_attribute_names = ['Cart Position', 'Cart Velocity', 'Pole Angle (in Radians)', 'Pole Angular Velocity']
for input_var_idx in range(4):
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    term_indices = antecedents_df.loc[antecedents_df['input_variable'] == input_var_idx]['term_index'].values
    for idx in term_indices:
        term_df = antecedents_df.query('input_variable == {} and term_index == {}'.format(input_var_idx, idx))
        if idx == selected_terms_for_each_input[input_var_idx]:
            fill_between = True
        else:
            fill_between = False
        plt_term(mins[input_var_idx], maxes[input_var_idx], term_df, gaussian_params,
                 '$G_{}^{}$'.format(input_var_idx, idx), styles[idx], fill_between)

    # plt.text(0.0, 0.9, '"$L$"', fontsize=30)
    # plt.text(0.325, 0.9, '"$A$"', fontsize=30)
    # plt.text(0.525, 0.9, '"$N$"', fontsize=30)
    # plt.text(0.725, 0.9, '"$E$"', fontsize=30)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    tick_size = ticks[input_var_idx]
    print(tick_size)
    # plt.xticks(np.arange(mins[input_var_idx], maxes[input_var_idx] + 1, tick_size))
    plt.xticks(rotation=45)
    plt.xlabel('({}) {}'.format(plt_lbls[input_var_idx], state_attribute_names[input_var_idx]), fontsize=20, labelpad=4,
               fontweight='bold')
    # plt.ylabel('$L(\hat{G}_{k}(x)) \in \mathscr{L}_{k}$', fontsize=30, labelpad=4)
    plt.ylabel('Degree of membership', fontsize=20, labelpad=4, fontweight='bold')
    offset = offsets[input_var_idx]
    # plt.legend(loc='upper left', bbox_to_anchor=[0.75 + offset, 0.5], frameon=False, prop={'weight': 'bold', 'size': 16})
    # plt.show()

    image_format = 'svg'  # e.g .png, .svg, etc.
    image_name = 'terms_{}.{}'.format(input_var_idx, image_format)
    fig.savefig(image_name, format=image_format, dpi=1200, bbox_images='tight')
