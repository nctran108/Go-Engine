import os
import tarfile
import gzip
import glob
import shutil

import numpy as np
from keras.utils import to_categorical

from go.gosgf import Sgf_game
from go.goboard import Board, GameState, Move
from go.gotypes import Player, Point
from go.encoders.base import get_encoder_by_name

from go.data.index_processor import KGSIndex
from go.data.sampling import Sampler

class GoDataProcessor:
    def __init__(self, encoder='oneplane', data_directory='data', size = 19):
        self.encoder = get_encoder_by_name(encoder, size)
        self.data_dir = os.getcwd() + '/go/' + data_directory
        self.size = size

    def load_go_data(self, data_type='train', num_samples=1000):
        index = KGSIndex(data_directory=self.data_dir+ '/raw')
        index.download_files()

        sampler = Sampler(data_dir=self.data_dir + '/raw')
        data = sampler.draw_data(data_type, num_samples)

        zip_names : (str) = set()
        indices_by_zip_name = {}
        for filename, index in data:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)
        
        processed_file_path = self.data_dir + "/processed"
        
        if not os.path.isdir(processed_file_path):
            os.makedirs(processed_file_path) # create the folder if not exist
        
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(processed_file_path + "/" + data_file_name):
                self.process_zip(zip_name, data_file_name, indices_by_zip_name[zip_name])
        
        features_and_labels = self.consolidate_games(data_type, data)
        return features_and_labels
        
    def process_zip(self, zip_file_name, data_file_name, game_list):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/raw/' + tar_file)
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list, name_list)

        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))

        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content.decode('utf-8'))

            game_state, first_move_done = self.get_handicap(sgf,sgf.get_size())

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col +1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    game_state = game_state.apply_move(move)
                    first_move_done = True
        
        feature_file_base = self.data_dir + '/processed/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/processed/' + data_file_name + '_labels_%d'

        chunk = 0 # Due to files with large content, split up after chunksize
        chunksize = 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content.decode('utf-8'))
                game_state, first_move_done = self.get_handicap(sgf,self.size)

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
    
    @staticmethod
    def get_handicap(sgf: Sgf_game, size):
        go_board = Board(size,size)
        first_move_done = False
        move = None
        game_state = GameState.new_game(size)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def consolidate_games(self, data_type, samples):
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)
        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/processed/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), self.size * self.size)
                feature_list.append(x)
                label_list.append(y)
        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        np.save('{}/features_{}.npy'.format(self.data_dir + '/processed/', data_type), features)
        np.save('{}/labels_{}.npy'.format(self.data_dir + '/processed/', data_type), labels)

        return features, labels
    
    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/raw/' + zip_file_name)  # <1>

        tar_file = zip_file_name[0:-3]  # <2>
        this_tar = open(self.data_dir + '/raw/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)  # <3>
        this_tar.close()
        return tar_file