import os
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
from multiprocessing import Pool, RLock, current_process, freeze_support, cpu_count
import time
from keras.utils import to_categorical
from tqdm import tqdm

from go.gosgf import Sgf_game
from go.goboard import Board, GameState, Move
from go.gotypes import Player, Point
from go.data.index_processor import KGSIndex
from go.data.sampling import Sampler
from go.data.generator import DataGenerator
from go.encoders.base import get_encoder_by_name
from go.utils import print_board

def worker(jobinfo):   
    try:
        i, clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(i,zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit) as e:
        print("From Worker: ",e)
        raise Exception('>>> Exiting child process.')

class GoDataProcessor:
    def __init__(self, encoder='simple', data_directory='data/raw'):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = os.getcwd() + '/go/' + data_directory
        self.num_samples = 0

    def generate_samples(self, data_type='train', num_samples=1000):
        self.num_samples = num_samples
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()
        
        print("[processor][generator] start sampler.....")
        sampler = Sampler(data_dir=self.data_dir,index=index)
        data = sampler.draw_data(data_type, num_samples)
        print("[processor][generator] data drawed....")
        return data

# tag::load_generator[]
    def load_go_data(self, data_type='train', num_samples=1000,
                     use_generator=False):
        self.num_samples = num_samples
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()
        
        print("[processor][load_data] start sampler.....")
        sampler = Sampler(data_dir=self.data_dir,index=index)
        data = sampler.draw_data(data_type, num_samples)
        print("[processor][load_data] data drawed....")

        self.map_to_workers(data_type, data)  # <1>
        if use_generator:
            generator = DataGenerator(data_type,self.data_dir, data)
            return generator  # <2>
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels  # <3>

# <1> Map workload to CPUs
# <2> Either return a Go data generator...
# <3> ... or return consolidated data as before.
# end::load_generator[]

    def unzip_data(self, zip_file_name):
        tar_file = zip_file_name[:-3]
        with gzip.open(self.data_dir + '/' + zip_file_name, 'rb') as this_gz, \
                open(self.data_dir + '/' + tar_file, 'wb') as this_tar:
            shutil.copyfileobj(this_gz, this_tar)
        return tar_file

    def process_zip(self, i, zip_file_name, data_file_name, game_list):
        pid = current_process().ident
        tqdm_text = "[pid " + "{}".format(pid).zfill(3) + ']'

        tar_file = self.unzip_data(zip_file_name)
        with tarfile.open(self.data_dir + '/' + tar_file) as zip_file:
            name_list = zip_file.getnames()
            total_examples = self.num_total_examples(zip_file, game_list, name_list)
            shape = self.encoder.shape()
            feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
            features = np.zeros(feature_shape)
            labels = np.zeros((total_examples,))

            counter = 0
            # hide the finished processing bar and start a save bar in the same position
            with tqdm(range(total_examples), desc=tqdm_text, position=i, leave=False) as process:
                for index in game_list:
                    name = name_list[index + 1]
                    if not name.endswith('.sgf'):
                        raise ValueError(name + ' is not a valid sgf')
                    sgf_content = zip_file.extractfile(name).read()
                    sgf = Sgf_game.from_string(sgf_content.decode('utf-8'))
                    
                    game_state, first_move_done = self.get_handicap(sgf)

                    for item in sgf.main_sequence_iter():
                        color, move_tuple = item.get_move()
                        point = None
                        move = None
                        if color is not None:
                            if move_tuple is not None:
                                row, col = move_tuple
                                point = Point(row + 1, col + 1)
                                move = Move.play(point)
                            else:
                                move = Move.pass_turn()
                            if first_move_done and point is not None:
                                features[counter] = self.encoder.encode(game_state)
                                labels[counter] = self.encoder.encode_point(point)
                                counter += 1
                                process.update(1) ## update bar
                            game_state = game_state.apply_move(move)
                            first_move_done = True

        save_desc = f"{tqdm_text} saving"
        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'
        chunk = 0  # Due to files with large content, split up after chunksize
        chunksize = 1024
        total_save_chunks = features.shape[0] // chunksize
        if total_save_chunks > 0:
            with tqdm(total=total_save_chunks, desc=save_desc, position=i, leave=False) as save_process:
                while features.shape[0] >= chunksize:
                    feature_file = feature_file_base % chunk
                    label_file = label_file_base % chunk
                    chunk += 1
                    current_features, features = features[:chunksize], features[chunksize:]
                    current_labels, labels = labels[:chunksize], labels[chunksize:]
                    np.save(feature_file, current_features)
                    np.save(label_file, current_labels)
                    save_process.update(1)
        else:
            while features.shape[0] >= chunksize:
                feature_file = feature_file_base % chunk
                label_file = label_file_base % chunk
                chunk += 1
                current_features, features = features[:chunksize], features[chunksize:]
                current_labels, labels = labels[:chunksize], labels[chunksize:]
                np.save(feature_file, current_features)
                np.save(label_file, current_labels)

        # Save any remaining data that didn't fill a full chunk
        if features.shape[0] > 0:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            np.save(feature_file, features)
            np.save(label_file, labels)

        # explicit cleanup after saving
        zip_file.close()
        #os.remove(self.data_dir + '/' + tar_file)
        #print(f"[processor][process_zip_file] finished processing {zip_file_name}")
        

    def consolidate_games(self, name, samples):
        print('[processor][consolidate] Start consoldate games.....')
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + name
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19 * 19)
                feature_list.append(x)
                label_list.append(y)

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        feature_file = self.data_dir + '/features_' + name
        label_file = self.data_dir + '/labels_' + name
        print('[processor][consolidate] start saving.....')
        np.save(feature_file, features)
        np.save(label_file, labels)
        print('[processor][consolidate] data stored......')

        return features, labels

    @staticmethod
    def get_handicap(sgf):  # Get handicap stones
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for coord in setup:
                    row, col = coord
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
                    move = Move(Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, game_state, move)
        return game_state, first_move_done

    def map_to_workers(self, data_type, samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)
        
    
        zips_to_process = []
        for i, zip_name in enumerate(zip_names):
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                zips_to_process.append((i,self.__class__, self.encoder_string, zip_name,
                                        data_file_name, indices_by_zip_name[zip_name]))
        
        cores = 8  # Determine number of CPU cores and split work load among them
        pool = Pool(cores, initargs=(RLock(),), initializer=tqdm.set_lock)

        p = pool.map_async(worker, zips_to_process)
        try:
            #async_results = [pool.apply_async(worker, (zip_to_process,)) for zip_to_process in zips_to_process]
            p.get()
            pool.close()
            pool.join()
        except (KeyboardInterrupt, TimeoutError, Exception) as e:  # Caught keyboard interrupt, terminating workers
            pool.terminate()
            pool.join()
            print("[processor][map_to_workers] Error")
            print("[processor][map_to_workers] " + str(type(e)))
            print("[processor][map_to_workers] " + str(e))
            exit(-1)


    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content.decode())
                game_state, first_move_done = self.get_handicap(sgf)
                
                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples
    
    def load_data_from_npy(self, data_type):
        print('[processor][load_data_from_npy] loading npy data....')
        feature_list = []
        label_list = []
        base = self.data_dir + '/' + '*_' + data_type + '_features_*.npy'
        for feature_file in tqdm(glob.glob(base)):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19 * 19)
                feature_list.append(x)
                label_list.append(y)

        if not feature_list:
            raise ValueError(f"No processed data files found for '{data_type}' in {self.data_dir}. Please run data processing first to generate the .npy files.")

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        process_dir = os.getcwd() + "/go/data/process"
        if not os.path.isdir(process_dir):
            os.makedirs(process_dir)

        feature_file = process_dir + '/features_' + data_type
        label_file = process_dir + '/labels_' + data_type
        print('[processor][load_data_from_npy] start saving.....')
        np.save(feature_file, features)
        np.save(label_file, labels)
        print('[processor][load_data_from_npy] data stored......')

        return features, labels