from go.agent import PolicyAgent
from go.agent.predict import load_prediction_agent
from go.encoders.alphago import AlphaGoEncoder
from go.RL.simulate import experience_simulation

import h5py

encoder = AlphaGoEncoder((19,19))

sl_agent = load_prediction_agent(h5py.File(''))
sl_opponent = load_prediction_agent(h5py.File(''))

alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
opponent = PolicyAgent(sl_opponent.model, encoder)

num_games = 1000
experience = experience_simulation(num_games, alphago_rl_agent, opponent)

alphago_rl_agent.train(experience)

with h5py.File('alphago_rl_policy.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience.h5', 'w') as exp_out:
    experience.serialize(exp_out)