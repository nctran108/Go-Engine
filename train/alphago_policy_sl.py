import sys
import os
sys.path.append(os.getcwd())

from go.data.parallel_processor import GoDataProcessor
from go.encoders.alphago import AlphaGoEncoder
from go.agent.predict import DeepLearningAgent
from AlphaZero import alpha_zero

from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np


rows, cols = 19, 19
num_classes = rows * cols
num_games = 10

encoder = AlphaGoEncoder((rows,cols))
processor = GoDataProcessor(encoder=encoder.name())

X, y = np.load('go/data/raw/features_train.npy')
test_X, test_y = np.load('go/data/raw/labels_train.npy')

input_shape = (encoder.num_planes, rows, cols)
alphago_sl_policy = alpha_zero.alphago_model(input_shape, is_policy_net=True)

alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

epochs = 200
batch_size = 128
alphago_sl_policy.fit(x,y, batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_X,test_y),
              callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.keras')])

alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
    alphago_sl_agent.serialize(sl_agent_out)
