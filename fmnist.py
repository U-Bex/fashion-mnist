import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator





epochs = 10
batch_size = 64
num_classes = 10
img_row, img_col, channel = 28, 28, 1
if sys.argv[1] == 't':
    augment_flag = True
elif sys.argv[1] == 'f':
    augment_flag = False
else:
    print('please ipnut t or f')
    exit(1)

print(augment_flag)

datagen = ImageDataGenerator(
            horizontal_flip = True,
            rotation_range = 30,
            zoom_range=0.2
            )

#input data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#data initialization
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#model define
model = Sequential()

model.add(Conv2D(32, (3,3),padding='same', activation = 'relu',input_shape=(img_row,img_col,channel)))
model.add(Conv2D(32, (3,3),padding='same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))
model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2),padding='same',))

model.add(Conv2D(128, (3,3),padding='same', activation = 'relu'))
model.add(Conv2D(128, (3,3),padding='same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

if not augment_flag:
    print('NOT using data augmentation')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
else:
    print('Using data augmentation')
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, subset='training'),
                        steps_per_epoch = len(x_train)/batch_size,
                        validation_data = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'),
                        validation_steps = 1,
                        epochs = epochs)
model_json = model.to_json()
open('fmnist_model1.json', 'w').write(model_json)
model.save_weights('fmnist_model1.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc :', score[1])

###graph plot#####
fig = plt.figure()
plt.plot(history.history['accuracy'], marker='.', label='train_acc')
plt.plot(history.history['val_accuracy'], marker='.', label='val_acc')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
#plt.show()
fig.savefig('./accuracy.png')
plt.close()

fig2 = plt.figure()
plt.plot(history.history['loss'], marker='.', label='train_loss')
plt.plot(history.history['val_loss'], marker='.', label='val_loss')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
#plt.show()
fig2.savefig('./loss.png')
plt.close()
