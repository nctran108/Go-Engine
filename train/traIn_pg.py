import sys
sys.path.append('d:\\study\\Go-Engine')
sys.path.append('d:\study\Go-Engine\go\RL')
import h5py
from go import agent
from go import RL

def main():
    learning_agent_filename = "large_model_sevenplane_encoder_with_SGD.h5"
    experience_files = []
    learning_rate = 0.0001
    clipnorm = 0.001
    batch_size = 512
    agent_out = "trained_pg_large_model_sevenplane_encoder_with_SGD.h5"
    learning_agent = agent.load_policy_agent(h5py.File(learning_agent_filename))
    for exp_filename in experience_files:
        exp_buffer = RL.load_experience(h5py.File(exp_filename))
        learning_agent.train(exp_buffer,
                             lr=learning_rate,
                             clipnorm=clipnorm,
                             batch_size=batch_size)
    with h5py.File(agent_out,'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == "__main__":
    main()