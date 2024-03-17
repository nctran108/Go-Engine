import h5py

from keras.models import Sequential
from keras.layers import Dense
from go.agent.predict import DeepLearningAgent, load_prediction_agent
from go.data.parallel_processor import GoDataProcessor
from go.encoders.sevenplane import SevenPlaneEncoder
from networks import large
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    go_board_size = 19
    num_classes = go_board_size * go_board_size
    encoder = SevenPlaneEncoder((go_board_size,go_board_size))
    processor = GoDataProcessor(encoder=encoder.name())

    generator = processor.load_go_data('train',num_samples=100,use_generator=True)
    generator_test = processor.load_go_data('test',num_samples=100,use_generator=True)
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
    
    epochs = 20
    batch_size = 128

    model.fit(generator.generate(batch_size,num_classes),
              epochs=epochs,
              steps_per_epoch=generator.get_num_samples() / batch_size,
              validation_data=generator_test.generate(batch_size,num_classes),
              validation_steps=generator_test.get_num_samples() / batch_size,
              callbacks=[ModelCheckpoint('../checkpoints/large_model_epoch_{epoch}.h5')]
              )
    """
    model.fit_generator(generator=generator.generate(batch_size,num_classes),
                epochs=epochs,
                steps_per_epoch=generator.get_num_samples() / batch_size,
                validation_data=generator_test.generate(batch_size,num_classes),
                validation_steps=generator_test.get_num_samples() / batch_size,
                callbacks=ModelCheckpoint('../checkpoints/large_model_epoch_{epoch}.h5'))
    """
    print("Finished fitting....")
            
    model.evaluate(generator_test.generate(batch_size,num_classes),
                   steps=generator_test.get_num_samples() / batch_size)

    deep_learning_bot = DeepLearningAgent(model, encoder)
    h5file = h5py.File("./go/agent/deep_bot.h5", 'w')
    deep_learning_bot.serialize( h5file)

    model_file = h5py.File("./go/agent/deep_bot.h5", 'r')
    bot_from_file = load_prediction_agent(model_file)
    