import sys
from pathlib import Path
# Get path for files
path_root = Path(__file__).resolve().parents[1] # Adjust the number
sys.path.append(str(path_root)+'/model')
import numpy as np
import math as mt
import time
import matplotlib.pyplot as plt
from neuralnetwork import NeuralNetwork

def test():
    inputSize = 100
    hiddenSize = 100 # nb of neurons per layer
    outputSize = 100
    nbHiddenLayer = 100
    
    mu = 0
    sigma = 0.1
    
    net = NeuralNetwork(inputSize, hiddenSize, outputSize, nbHiddenLayer)
    
    weights = np.random.normal(mu, sigma, net.numberSynapses)
    net.setWeights(weights)

    X = np.random.normal(0, 1, inputSize)
        
    print("Input")
    print(X)
    print('Simulation')
    outputs = net.run(X, option='positive', debug=True)

def percolation_sigmas(mu=-0.5, sigmas = [0.1, 1.5, 5], nbTrial=1):
    inputSize = 100
    hiddenSize = 100 # nb of neurons per layer
    outputSize = 100
    nbHiddenLayer = 100
    net = NeuralNetwork(inputSize, hiddenSize, outputSize, nbHiddenLayer)
    
    fig, axes = plt.subplots(1, len(sigmas), constrained_layout=True,
                             figsize=(15, 6))
    
    # Interesting case:
    for i, sigma in enumerate(sigmas):
        seed = 1900
        np.random.seed(seed)
        # Initialisation
        weights = np.random.normal(mu, sigma, net.numberSynapses)
        net.setWeights(weights)
        X = np.heaviside(np.random.uniform(-1, 1, inputSize), 0)
        
        # Run
        net.run(X, option='step') 
        
        # states matrix
        inputs = net.inputs
        hiddenLayer = np.vstack((net.hiddenLayer))
        matrix_states = np.vstack((inputs, hiddenLayer))
        matrix_states = np.vstack((matrix_states, net.outputs))
        
        x = np.arange(0, 100)
        y = np.arange(0, 100)
        X, Y = np.meshgrid(x, y)
        
        # Weight matrix
        # W = net.W
        ax = axes[i]
        ax.set_title(f'$\mu$={mu}, $\sigma$={sigma}')
        ax.imshow(matrix_states,interpolation='none',cmap=plt.cm.gist_gray, origin='lower')  
        ax.set_xlabel('Neuron state x')
        ax.set_ylabel('Index of the layer')
    plt.show()

def percolation_seed(mu=-0.5, sigma = 1.5,     
                     seeds = [1700, 1600, 1900, 1400]):
    inputSize = 100
    hiddenSize = 100 # nb of neurons per layer
    outputSize = 100
    nbHiddenLayer = 100
    net = NeuralNetwork(inputSize, hiddenSize, outputSize, nbHiddenLayer)
    
    fig, axes = plt.subplots(1, len(seeds), constrained_layout=True,
                             figsize=(15, 6))
    fig.suptitle(f'$\mu$={mu}, $\sigma$={sigma}')

    # Interesting case:
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        # Initialisation
        weights = np.random.normal(mu, sigma, net.numberSynapses)
        net.setWeights(weights)
        X = np.heaviside(np.random.uniform(-1, 1, inputSize), 0)
        
        # Run
        net.run(X, option='step') 
        
        # states matrix
        inputs = net.inputs
        hiddenLayer = np.vstack((net.hiddenLayer))
        matrix_states = np.vstack((inputs, hiddenLayer))
        matrix_states = np.vstack((matrix_states, net.outputs))
        
        x = np.arange(0, 100)
        y = np.arange(0, 100)
        X, Y = np.meshgrid(x, y)
        
        # Weight matrix
        # W = net.W
        ax = axes[i]
        ax.set_title(f'seed={seed}')
        ax.imshow(matrix_states,interpolation='none',cmap=plt.cm.gist_gray, origin='lower')  
        ax.set_xlabel('Neuron state x')
        ax.set_ylabel('Index of the layer')
    plt.show()

def simulation():
    
    # Network    
    inputSize = 100
    outputSize = 100
    hiddenSize = 100 # nb of neurons per hidden layer
    nbHiddenLayer = 100
    net = NeuralNetwork(inputSize, hiddenSize, outputSize, nbHiddenLayer)
    
    # Simulation parameters
    mu = -0.5
    sigmas = np.linspace(0.01, 10, 1000)
    nbTrial = 10
    
    avgActivity = []
    varActivity = []
    for sigma in sigmas:
        avgBuff = []
        varBuff = []
        for k in range(nbTrial):
              
            # Initialisation
            weights = np.random.normal(mu, sigma, net.numberSynapses)
            net.setWeights(weights)
            X = np.heaviside(np.random.uniform(-1, 1, inputSize), 0)            
            
            # Run
            net.run(X, option='step') 
            
            # Save trials
            avgBuff.append(np.mean(net.outputs))
    
        # Saving data
        avgActivity.append(np.mean(avgBuff))
        varActivity.append(np.var(avgBuff))  
        
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r'$\sigma(W)$')
    ax.set_ylabel('$<Y>$')
    ax.set_title(f'Output mean, averaged over {nbTrial} trials')
    ax.plot(sigmas, avgActivity, '.-')
    
    fig1 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel(r'$\sigma(W)$')
    ax1.set_ylabel('$\sigma^2(<Y>)_k$')
    ax1.set_title(f'Variance of outputs, averaged over {nbTrial} trials')
    ax1.plot(sigmas, varActivity, '.-')
    plt.show()

    
def main():
    simulation()

if __name__ == '__main__':
    time_start = time.perf_counter()
    main()
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in '+str(time_elapsed)+'s')