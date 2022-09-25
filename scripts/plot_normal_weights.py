# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:59:59 2022

@author: Manu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
 
rcParams['font.size'] = 18

mu = -0.5
sigma = 1
W = np.random.normal(mu, sigma, 10000)
plt.hist(W, bins=100, density=True)
plt.xlabel('W')
plt.ylabel('P(W)')
plt.title('Histogram of weights')
plt.tight_layout()