# MADDPG
Use Multi Agents Deep Deterministic Policy Gradient(MADDPG) Algorithm to Find Collision-free Paths for Ships

## Known dependencies: 
  Python: 3.9  
  CUDA: 11.2  
  
  (Python package:)  
  pytorch: 1.10.2  
  tensorboard: 2.9.0  
  numpy: 1.21.5  
  matplotlib: 3.4.1  
  ffmpeg: 2.7.0  
  os, math, random: Python built-in package

## Known issues:
  1. Networks of agents in this project are simple. That is why GPU acceleration technology(CUDA 11.2, GPU: GTX965m) is used in training process, but the profermance is not obvious. On the other side, increasing the number of neurons, which will give significant GPU acceleration, may cause other problems like a too big hidden layer. In this case, networks would only output boundary values of the last activated function.
