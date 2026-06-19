import sys
import os
sys.path.append(os.getcwd())

from go.agent.policy import PolicyAgent
from go.agent.predict import load_prediction_agent
from go.RL.simulate import experience_simulation
import h5py

sl_agent = load_prediction_agent(h5py.File('alphago_sl_policy.h5'))
sl_opponent = load_prediction_agent(h5py.File('alphago_sl_policy.h5'))

alphago_rl_agent = PolicyAgent(sl_agent.model, sl_agent.encoder)
opponent = PolicyAgent(sl_opponent.model, sl_opponent.encoder)

num_games = 10
num_train = 1000
i = 0
while i < num_train:
    experience = experience_simulation(num_games, alphago_rl_agent, opponent)

    alphago_rl_agent.train(experience)
    i += 1
    if i % 10 == 0:
        print(f'Completed {i} games')
        with h5py.File('rl_agents/alphago_rl_policy_{i}.h5', 'w') as rl_agent_out:
            alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_policy.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience.h5', 'w') as exp_out:
    experience.serialize(exp_out)