from go.agent.base import Agent
from go import kerasutil
from go import encoders
from go import goboard
from go.agent.helpers import is_point_an_eye

import numpy as np
from keras import backend as k
from keras.optimizers import SGD

class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self._model = model
        self._encoder = encoder
        self._collector = None
        self._temperature = 0.0

    def set_collector(self, collector):
        self._collector = collector


    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height

        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def select_move(self, game_state):
        # encode a game state as a numerical tensor, then pass that tensor to my model to get move probabilities.
        board_tensor = self._encoder.encode(game_state)
        X = np.array([board_tensor])

        if np.random.random() < self._temperature:
            # Explore random moves.
            move_probs = np.ones(num_moves) / num_moves
        else:
            # the keras predict call makes batch predictions, so wrap the single board in an array and pull out the first
            # item from the resulting array
            move_probs = self._model.predict(X)[0]

        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # creates an array containing the index of every point on the board
        num_moves = self._encoder.board_width * self._encoder.board_height
        candidates = np.arange(num_moves)
        # samples from the points on the board according to the polocy, creates a ranked list of points to try
        ranked_moves = np.random.choice(candidates, num_moves,
                                        replace=False, p=move_probs)
        
        # loops over each point, checks if it's valid move, and picks the first valid one
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player)
            if is_valid and (not is_an_eye):
                if self._collector is not None:
                    self._collector.record_decision(state=board_tensor,
                                                    action=point_idx)
                return goboard.Move.play(point)
        # if fall through here, there are no reasonable moves left.
        return goboard.Move.pass_turn()
    
    def train(self, experience, learning_rate, clipnorm, batch_size):
        self._model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=learning_rate, clipnorm=clipnorm))
        
        experience_size = experience.states.shape[0]
        num_moves = self._encoder.board_width * self._encoder.board_height
        y = np.zeros((experience_size, num_moves))
        for i in range(experience_size):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward
        
        self._model.fit(experience.states, y,
                        batch_size=batch_size,
                        epochs=1)

def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(
        h5file['model'],
        custom_objects={'policy_gradient_loss': policy_gradient_loss})
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width,board_height))
    return PolicyAgent(model, encoder)

def policy_gradient_loss(y_true, y_pred):
    clip_pred = k.clip(y_pred, k.epsilon(), 1  - k.epsilon())
    loss = -1 * y_true * k.log(clip_pred)
    return k.mean(k.sum(loss, axis=1))