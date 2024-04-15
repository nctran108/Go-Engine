from go.data.parallel_processor import GoDataProcessor
from go.encoders.alphago import AlphaGoEncoder
from multiprocessing import freeze_support
import numpy as np

rows, cols = 19, 19
encoder = AlphaGoEncoder((rows,cols))
processor = GoDataProcessor(encoder=encoder.name())

def generate_sample(data_type='train', num_games=1000):
    #samples = processor.generate_samples(data_type, num_games)
    #processor.map_to_workers(data_type, samples)
    processor.load_data_from_npy(data_type)

def main():
    num_games = 1000
    # generate train
    #generate_sample('train', num_games)
    # generate test
    generate_sample('test', num_games)

if __name__ == '__main__':
    freeze_support() # support window
    main()
