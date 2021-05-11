from keras.preprocessing.image import ImageDataGenerator
from architecture import *
import matplotlib.pyplot as plt

train_dir = r'C:\Users\aishw\PycharmProjects\pythonProject\cell_images\train'
validation_dir = r'C:\Users\aishw\PycharmProjects\pythonProject\cell_images\val'

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(validation_dir,
                                                       target_size=(150, 150),
                                                       batch_size=20,
                                                       class_mode='categorical')

print("Image preprocessing complete")


model = malaria_net(input_shape=(150, 150, 3))

r = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(validation_generator))

print(r.history.keys())

model.save('basic_malaria_pos_neg_v1.h5')

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('LossVal_loss_vgg')


plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('AccVal_acc_vgg')