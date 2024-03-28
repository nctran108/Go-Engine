import h5py

from keras.models import Sequential
from keras.layers import Dense
from go.agent.predict import DeepLearningAgent
from go.data.parallel_processor import GoDataProcessor
from go.encoders.sevenplane import SevenPlaneEncoder
from networks import large
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    go_board_size = 19
    num_classes = go_board_size * go_board_size
    encoder = SevenPlaneEncoder((go_board_size,go_board_size))
    processor = GoDataProcessor(encoder=encoder.name())

    x,y = processor.load_go_data(num_samples=10000)
    x_test, y_test = processor.load_go_data('test',num_samples=1000)
    print("Got features and layers")

    input_shape = (encoder.num_planes, go_board_size, go_board_size)
    
    network_layers = large.layers(input_shape=input_shape)
    model = Sequential()

    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    epochs = 100
    batch_size = 128

    model.fit(x,y, batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test,y_test),
              callbacks=[ModelCheckpoint('./checkpoints/large_model_epoch_{epoch}.keras')]
              )
            
    score = model.evaluate(x_test,y_test,
                   verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    deep_learning_bot = DeepLearningAgent(model, encoder)
    h5file = h5py.File("./go/agent/deep_bot.h5", 'w')
    deep_learning_bot.serialize(h5file)

    