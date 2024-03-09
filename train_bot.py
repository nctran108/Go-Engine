import h5py

from keras.models import Sequential
from keras.layers import Dense
from go.agents.predict import DeepLearningAgent, load_prediction_agent
from go.data.parallel_processor import GoDataProcessor
from go.encoders.sevenplane import SevenPlaneEncoder
from networks import large

if __name__ == '__main__':
    go_board_size = 19
    num_classes = go_board_size * go_board_size
    encoder = SevenPlaneEncoder((go_board_size,go_board_size))
    processor = GoDataProcessor(encoder=encoder.name())

    X , y = processor.load_go_data(num_samples=100)
    print("Got features and layers")

    input_shape = (encoder.num_planes, go_board_size, go_board_size)
    
    model = Sequential()
    network_layers = large.layers(input_shape=input_shape)

    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    model.fit(X, y,
            batch_size=128,
            epochs=20,
            verbose=1)

    deep_learning_bot = DeepLearningAgent(model, encoder)
    h5file = h5py.File("./go/agents/deep_bot.h5", 'w')
    deep_learning_bot.serialize(h5file)

    model_file = h5py.File("./go/agents/deep_bot.h5", 'r')
    bot_from_file = load_prediction_agent(model_file)
    