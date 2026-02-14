from go.data.parallel_processor import GoDataProcessor
from go.encoders.alphago import AlphaGoEncoder
from multiprocessing import freeze_support
import numpy as np

rows, cols = 19, 19
encoder = AlphaGoEncoder((rows,cols))
processor = GoDataProcessor(encoder=encoder.name())

def generate_train_sample(num_games=1000):
    samples = processor.generate_samples('train', num_games)
    processor.map_to_workers('train', samples)
    processor.load_data_from_npy('train')

def generate_test_sample(num_games=100):
    samples = processor.generate_samples('test', num_games)
    processor.map_to_workers('test', samples)
    processor.load_data_from_npy('test')

def main():
    num_games = 1000
    # generate train
    # generate_train_sample(num_games)
    # generate test
    generate_test_sample()

if __name__ == '__main__':
    freeze_support() # support window
    main()
