import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout

# 1 load  and processing datas
np.random.seed(123)
X = np.load('generated_games/features-40k.npy')
Y = np.load('generated_games/labels-40k.npy')

samples = X.shape[0]
board_size = 9
input_shape = (board_size, board_size, 1)

X = X.reshape(samples, board_size, board_size, 1)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

# 2 generate models with 3 dense
model = Sequential()
model.add(Conv2D(filters=48, # this is the output not input
                 kernel_size=(3,3),
                 activation='relu',
                 padding='same',
                 input_shape=input_shape))
model.add(Dropout(rate=0.5))

model.add(Conv2D(48, (3,3),
                 padding='same',
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(board_size * board_size, activation='softmax'))
model.summary()

# 3: model complile
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 4: model fitness and evaluation
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=100,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])