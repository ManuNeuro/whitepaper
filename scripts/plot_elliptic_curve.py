# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:45:40 2022

@author: Manu
"""
import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/19756043/python-matplotlib-elliptic-curves
# https://jeremykun.com/2014/02/24/elliptic-curves-as-python-objects/

def elliptic_curve(x, y, a=-1, b=1):
    # secp256k1
    return pow(y, 2) - pow(x, 3) - x * a - b

delta = 0.01
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

P = elliptic_curve(X, Y)

plt.contour(X, Y, P, [0])

