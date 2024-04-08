from go.data.parallel_processor import GoDataProcessor
from go.encoders.alphago import AlphaGoEncoder

rows, cols = 19, 19
num_classes = rows * cols
num_games = 10000

encoder = AlphaGoEncoder((rows,cols))
processor = GoDataProcessor(encoder=encoder.name())

# generate train
samples = processor.generate_samples('train', num_games)
processor.map_to_workers('train', samples)
processor.load_data_from_npy('train')

# generate test
samples = processor.generate_samples('test', num_games)
processor.map_to_workers('test', samples)
processor.load_data_from_npy('test')