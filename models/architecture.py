from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers


def malaria_net(input_shape=(150, 150, 3), loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5),
                metrics=['acc']):
    model = Sequential(name='Malaria_Net')
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics
                  )

    model.summary()
    print("Malaria_Net Created Successfully!")

    return model
