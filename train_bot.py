import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout
from go.agent.predict import DeepLearningAgent
from go.data.parallel_processor import GoDataProcessor
from go.encoders.sevenplane import SevenPlaneEncoder
from networks import large
from keras.callbacks import ModelCheckpoint
import numpy as np

if __name__ == '__main__':
    board_size = 19
    num_classes = board_size * board_size
    encoder = SevenPlaneEncoder((board_size,board_size))
    processor = GoDataProcessor(encoder=encoder.name())

    features = np.load('go/data/raw/features_train.npy')
    labels = np.load('go/data/raw/labels_train.npy')

    random_indices = np.random.randint(0, 5000, size=1000)

    x, x_test = features[:5000], features[random_indices]
    y , y_test = labels[:5000], labels[random_indices]

    input_shape = (encoder.num_planes, board_size, board_size)
    
    network_layers = large.layers(input_shape=input_shape)
    model = Sequential()

    for layer in network_layers:
        model.add(layer)
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    epochs = 300
    batch_size = 16

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

    