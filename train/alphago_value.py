import sys
import os
sys.path.append(os.getcwd())

from networks.alphaGo import alphaGoModel
from go.encoders.alphago import AlphaGoEncoder
from go.RL import ValueAgent, load_experience
import h5py

rows,cols = 19,19
encoder = AlphaGoEncoder((rows,cols))

input_shape = (encoder.num_planes, rows, cols)

alphago_value_network = alphaGoModel(input_shape)

alphago_value = ValueAgent(alphago_value_network, encoder)

experience = load_experience(h5py.File('rl_agents/experience_10.h5', 'r'))

alphago_value.train(experience)

with h5py.File('alphago_value.h5', 'w') as value_agent_out:
    alphago_value.serialize(value_agent_out)
