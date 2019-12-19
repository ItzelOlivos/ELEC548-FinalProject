import numpy as np
import pandas as pd
import sys

maze_type = sys.argv[1]
MAZE_SIZE = 220

def compute_muc(maze, maze_size, num):
    if maze == 's':
        mu = np.asarray([np.random.uniform(0, maze_size, 2) for i in range(num)])
        return mu
    elif maze == 'c':
        mu = []
        for i in range(num):
            r = np.random.uniform(0, maze_size / 2)
            theta = np.random.uniform(0, 2 * np.pi)
            mu.append([r * np.cos(theta), r * np.sin(theta)])
        return np.asarray(mu)

def compute_muh(num):
    mu = np.asarray([np.random.uniform(-np.pi, np.pi) for i in range(num)])
    return mu

N = 256
M = 12
K = 20
s = 0.01 * 100

muc = compute_muc(maze_type, MAZE_SIZE, N)
muh = compute_muh(M)

np.save('data/square_env_chunks/{}_mu_c'.format(maze_type), muc)
np.save('data/square_env_chunks/{}_mu_h'.format(maze_type), muh)