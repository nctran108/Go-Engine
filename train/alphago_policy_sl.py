import sys
import os
sys.path.append(os.getcwd())

from go.data.parallel_processor import GoDataProcessor
from go.encoders.alphago import AlphaGoEncoder
from go.agent.predict import DeepLearningAgent
from AlphaZero import alpha_zero

from keras.callbacks import ModelCheckpoint
import h5py

rows, cols = 19, 19
num_classes = rows * cols
num_games = 10000

encoder = AlphaGoEncoder((rows,cols))
processor = GoDataProcessor(encoder=encoder.name())
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)

input_shape = (encoder.num_planes, rows, cols)
alphago_sl_policy = alpha_zero.alphago_model(input_shape, is_policy_net=True)

alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

epochs = 200
batch_size = 128
alphago_sl_policy.fit(generator.generate(batch_size, num_classes),
                      epochs=epochs,
                      steps_per_epoch=generator.get_num_samples(batch_size,num_classes) / batch_size,
                      validation_data=test_generator.generate(batch_size,num_classes),
                      validation_steps=test_generator.get_num_samples(batch_size,num_classes) / batch_size,
                      callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.keras')])

alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
    alphago_sl_agent.serialize(sl_agent_out)
