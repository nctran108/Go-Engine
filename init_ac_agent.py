import argparse

import h5py

import networks
from go.agent.actor_critic import ACAgent
from go import encoders
import networks.two_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('--network', default='large')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name('sevenplane', args.board_size)

    model = networks.two_output.layers(encoder)

    new_agent = ACAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()