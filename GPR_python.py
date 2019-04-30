#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:49:01 2019

@author: Abbas
"""
# =============================================================================
# import
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as WK,\
ExpSineSquared as ESS, RationalQuadratic as RQ, Matern as M

# =============================================================================
# help
# =============================================================================
def help_GP():
    '''
    In the case of Gaussian process regression, 
    the system response is assumed to be a realization from a Gaussian process
    In the followiing fihures, Overlaid colored lines represent five realizations of the Gaussian process.
    
    Ref: Probabilistic Machine learning for Civil Engineers, J. Goulet. 2018 
    The code has been written by Abbas SheikhMohammadZadeh and Modified by M.R. Moeini. 
    
    Instruction:
        Write your date (observation) as a vector (for x and y).
    
    '''

# =============================================================================
# Write your data here (observations)
# =============================================================================
X_obs = np.array(np.array([90.27, 6.21, 2.23, 35.54, 82.88, 29.54, 38.65, 93.32, 97.29]), ndmin=2).T
f_obs = np.array([10.03, 6.43, 6.10, 5.89, 7.98, 5.99, 6.10, 10.67, 11.03])


# =============================================================================
# Function to show the summary of the fit
# =============================================================================
def summary(gp):
    optimized = gp.optimizer != None
    if not optimized:
        s1 = "Fitted Kernel(not optimized)\n\n%s" % gp.kernel_
    else:
        s1 = "Fitted Kernel(Optimized)\n\n%s" % gp.kernel_
    s2 = "\n\nlog marginal likelihood: %.5f" % gp.log_marginal_likelihood(gp.kernel_.theta)
    print(s1 + s2 + '\n')
    

# =============================================================================
# Gaussian process regression
# =============================================================================
# Specify a kernel
kernel = 1 * RBF(1, (1e-2, 1e2))
gp = GPR(kernel=kernel, alpha = 0, n_restarts_optimizer=9)

# Fit to data & optimize hyperparameters w.r.t. maximizing marginal likelihood
gp.fit(X_obs, f_obs)
summary(gp)

# Make a prediction on several test points
X_test = np.array(np.linspace(0, 100, 100), ndmin = 2).T
f_mean, f_var = gp.predict(X_test, return_std=True)

# Create a figure
fig_noise_free = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 20})

# Mark the observations
plt.plot(X_obs, f_obs, 'ro', label='observations')

# Draw a mean function and 95% confidence interval
plt.plot(X_test, f_mean, 'b-', label='mean function')
upper_bound = f_mean + 1.96 * f_var
lower_bound = f_mean - 1.96 * f_var
plt.fill_between(X_test.ravel(), lower_bound, upper_bound, color = 'b', alpha = 0.1,
                 label='95% confidence interval')

# Draw samples from the posterior and plot
X_samples = np.array(np.linspace(0, 100, 100), ndmin = 2).T
seed = np.random.randint(10) # random seed
plt.plot(X_samples, gp.sample_y(X_samples, n_samples = 5, random_state = seed), ':')

# Plot
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$f(x)$', fontsize=20)
plt.xlim(X_test.min(), X_test.max())
plt.legend(loc='upper left')
plt.title('A GP posterior with Noise-free observations')
plt.show()


# =============================================================================
# Gaussian process regression  -  Noisy
# =============================================================================
kernel = 1 * RBF(1, (1e-2, 1e2)) + WK(3)
gp = GPR(kernel=kernel, alpha = 0, n_restarts_optimizer=10)
gp.fit(X_obs, f_obs)
summary(gp)

# Make a prediction on test points
X_test = np.array(np.linspace(0, 100, 100), ndmin = 2).T
f_mean, f_var = gp.predict(X_test, return_std=True)

# Create a Plot
fig_noisy = plt.figure(figsize = (20,12))
plt.rcParams.update({'font.size': 20})

# Mark the observations
plt.plot(X_obs, f_obs, 'ro', label='observations')

# Draw a mean function and 95% confidence interval
plt.plot(X_test, f_mean, 'b-', label='mean function')
upper_bound = f_mean + 1.96 * f_var
lower_bound = f_mean - 1.96 * f_var
plt.fill_between(X_test.ravel(), lower_bound, upper_bound, color = 'b', alpha = 0.1,
                 label='95% confidence interval')

# Draw samples from the posterior and plot
X_samples = np.array(np.linspace(0, 100, 100), ndmin = 2).T
seed = np.random.randint(10) # random seed
plt.plot(X_samples, gp.sample_y(X_samples, n_samples = 10, random_state = seed), ':')

# Plot
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.xlim(X_test.min(), X_test.max())
plt.legend(loc='upper left')
plt.title('A GP posterior with Noisy observations(RBF Kernel)')
plt.show()