import tensorflow as tf
import numpy as np

# --------------------------- CONSTANTS ---------------------------
# Ground Truth Computation
N = 256
M = 12
K = 20
s = 0.01 * 100

# Maze Exploration
MAZE_SIZE = 220
MAZE_TYPE = 's'
EPISODES = 914
STEPS_PER_EPISODE = 700

# Training
BATCH_SIZE = 100
TRAINING_STEPS = int(EPISODES*.70)*STEPS_PER_EPISODE
HIDDEN_UNITS = 128
LEARNING_RATE = 10e-5
GRAD_CLIPPING_TRES = 10e-5

# Neural activity
GRID_RESOLUTION = 30
OUTPUT_NAME = 'rates_v1'

# ------------------------- DATA ACQUISITION ------------------------
data = np.zeros(275)

for idx in range(10):
    data = np.vstack(
        (data, np.load('data/proc-rows/square_data_chunk_{}_proc.npy'.format(idx), allow_pickle=True, encoding='latin1')))
    print (idx)

data = np.delete(data, (0), axis=0)
data = data[:EPISODES*STEPS_PER_EPISODE]

print('Shape of data: {}'.format(data.shape))

muc = np.load('data/centers/{}_mu_c.npy'.format(MAZE_TYPE))
muh = np.load('results_big/{}_mu_h.npy'.format(MAZE_TYPE))

indexes_place_cell = [i for i in range(7, 7+N)]
indexes_head_cell = [i for i in range(7+N, 7+N+M)]
sim_len = EPISODES*STEPS_PER_EPISODE

# ------------------------- DATA SPLITTING ------------------------
def multivariate_chunk(dataset, target1, target2, sim_paths, start_index, end_index, history_size):
    data = []
    labels_place = []
    labels_direction = []
    true_path = []

    indices = [i for i in range(start_index, end_index + history_size, history_size)]

    for idx in range(len(indices) - 1):
        data.append(dataset[indices[idx]:indices[idx + 1]])
        labels_place.append(target1[indices[idx]:indices[idx + 1]])
        labels_direction.append(target2[indices[idx]:indices[idx + 1]])
        true_path.append(sim_paths[indices[idx]:indices[idx + 1]])

    return np.asarray(data, np.float32), np.asarray(labels_place, np.float32), np.asarray(labels_direction,
                                                                                          np.float32), np.asarray(
        true_path, np.float32)


train_samples, train_labels_place, train_labels_direction, train_paths = multivariate_chunk(data[:, [0, 1, 6]],
                                                                                            data[:, indexes_place_cell],
                                                                                            data[:, indexes_head_cell],
                                                                                            data[:, [2, 3]],
                                                                                            0, TRAINING_STEPS,
                                                                                            BATCH_SIZE)

test_samples, test_labels_place, test_labels_direction, test_paths = multivariate_chunk(data[:, [0, 1, 6]],
                                                                                        data[:, indexes_place_cell],
                                                                                        data[:, indexes_head_cell],
                                                                                        data[:, [2, 3]],
                                                                                        TRAINING_STEPS, data.shape[0],
                                                                                        BATCH_SIZE)

print('Data splitting is complete. ')

# ------------------------- NEURAL NETWORK ARCHITECTURE ------------------------
def compute_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM((HIDDEN_UNITS),
                                   batch_input_shape=(None, None, train_samples.shape[2]),
                                   return_sequences=True,
                                   name='LSTM'))

    model.add(tf.keras.layers.Dropout(0.5, name='linear'))

    body = model.get_layer(name='linear')

    out1 = tf.keras.layers.Dense(M, activation='softmax', name='dir')(body.output)

    out2 = tf.keras.layers.Dense(N, activation='softmax', name='place')(body.output)

    composed_model = tf.keras.models.Model(inputs=model.input, outputs=[out1, out2])

    RMSprop = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE, clipvalue=GRAD_CLIPPING_TRES)

    composed_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=RMSprop)

    return composed_model

model = compute_model()

print('Model summary')
print(model.summary())

history = model.fit(train_samples, [train_labels_direction, train_labels_place], epochs=10,
                    validation_data=(test_samples, [test_labels_direction, test_labels_place]))

# Save history to plot in mac

# ------------------------- COMPUTE NEURAL ACTIVITY RATES ------------------------
linear_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(index=1).output)
linear_layer_output = linear_layer_model.predict(train_samples)

resolution = GRID_RESOLUTION
maze_extents = MAZE_SIZE
rate = np.zeros([HIDDEN_UNITS, resolution, resolution])

for i in range(linear_layer_output.shape[0]):
    activation = np.maximum(linear_layer_output[i, :, :], 0)

    for h in range(HIDDEN_UNITS):
        for j in range(BATCH_SIZE):
            x = (train_paths[i, j, 0] / maze_extents) * resolution
            y = (train_paths[i, j, 1] / maze_extents) * resolution
            rate[h, int(x), int(y)] += activation[j, h]

    if np.mod(i, 1000) == 0:
        print(i)

np.save(OUTPUT_NAME, rate)