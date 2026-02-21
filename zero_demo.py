import sys
import os
import signal
sys.path.append(os.getcwd())

from go.encoders import ZeroEncoder
from go.agent import ZeroAgent, load_zero_agent
from go.RL import ZeroExperienceCollector, combine_zero_experience
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model

from go.gotypes import Player
from go.goboard import GameState, Player
from go.score import compute_game_result

game_played = 0
CONTROL_C = False

def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = compute_game_result(game)
    print(game_result)
    # Give the reward to the right agent.
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)

def signal_handler(sig, frame):
    global CONTROL_C
    print('CTRL-C was pressed')
    CONTROL_C = True

def main():
    global CONTROL_C
    global game_played
    signal.signal(signal.SIGINT, signal_handler)
    board_size = 9
    encoder = ZeroEncoder(board_size)
    
    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input
    
    layers = 40
    for i in range(layers): 
        pb = Conv2D(64, (3, 3),
                padding='same', 
                data_format='channels_first',
                activation='relu')(pb)
    
    policy_conv = Conv2D(2, (1, 1), 
        data_format='channels_first',
        activation='relu')(pb)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(encoder.num_moves(), activation='softmax')(policy_flat)
    

    value_conv = Conv2D(1, (1, 1),
        data_format='channels_first',
        activation='relu')(pb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(inputs=[board_input],
                outputs=[policy_output, value_output])
    rounds = 1600
    black_agent = ZeroAgent(model, encoder, rounds_per_move=rounds, c=2.0)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=rounds, c=2.0)
    
    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    num_games = 5
    
    for i in range(num_games):
        simulate_game(board_size, black_agent, c1, white_agent, c2)
        game_played += 1
        if CONTROL_C:
            break

    exp = combine_zero_experience([c1, c2])

    black_agent.train(exp, 0.01, 2048)

    black_agent.serialize(f'bots/zero_demo_{board_size}x{board_size}_{game_played}_games_{rounds}.weights.h5', json_file=True)     

if __name__ == "__main__":
    main()