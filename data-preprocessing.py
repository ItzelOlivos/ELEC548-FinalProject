import numpy as np
import pandas as pd
import sys

maze_type = sys.argv[1]
git_url = sys.argv[2]

data = pd.read_csv(git_url)

N = 256
M = 12
K = 20
s = 0.01 * 100

muc = np.load('data/square_env_chunks/{}_mu_c.npy'.format(maze_type))
muh = np.load('data/square_env_chunks/{}_mu_h.npy'.format(maze_type))

for n in range(N):
    data['place_cell_{}'.format(n)] = None

for m in range(M):
    data['head_direction_cell_{}'.format(m)] = None

name_columns = data.columns

names_place_cell = [name_columns[i] for i in range(7, 7 + N)]
names_head_cell = [name_columns[i] for i in range(7 + N, 7 + N + M)]

for ptr in range(data.shape[0]):

    x = data[['x', 'y']].iloc[ptr].values

    a1 = np.asarray([np.exp(-(np.linalg.norm(x - muc[i]) ** 2) / (2 * s ** 2)) for i in range(N)])
    b1 = np.sum(a1)

    try:
        data.at[ptr, names_place_cell] = a1 / b1
    except:
        data.at[ptr, names_place_cell] = np.zeros([len(a1)])

    h = []
    phi = data['direction_global'].iloc[ptr]

    a2 = np.asarray([np.exp(K * np.cos(phi - muh[i])) for i in range(M)])
    b2 = np.sum(a2)

    try:
        data.at[ptr, names_head_cell] = a2 / b2
    except:
        data.at[ptr, names_head_cell] = np.zeros([len(a2)])

    if np.mod(ptr, 1000) == 0:
        print(ptr)

res = data.to_numpy()
np.save(git_url.replace('.csv', '_proc'), res)