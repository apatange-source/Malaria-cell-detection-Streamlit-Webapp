import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # this converts it into RGB
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 3)
    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_gradcam(model, image_path):
    model_copy = keras.models.clone_model(model)
    img = get_img(image_path)

    model_copy.summary()

    model_copy.layers[-1].activation = None

    preds = model.predict(img)
    pred = np.rint(preds)
    heatmap = make_gradcam_heatmap(img, model, 'conv2d_3')
    if pred[0][0] == 1:
        return 'Parasitized', heatmap
    elif pred[0][1] == 1:
        return 'Uninfected', heatmap


def display_gradcam(img_path, heatmap, alpha=1):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


if __name__ == '__main__':
    model = tf.keras.models.load_model('basic_malaria_pos_neg_v1.h5')
    image_path = r'C:\Users\aishw\PycharmProjects\pythonProject\cell_images\test\Parasitized\C39P4thinF_original_IMG_20150622_105803_cell_91.png'
    pred, heatmap = apply_gradcam(model, image_path)
    print(pred)
    super_imposed = display_gradcam(image_path, heatmap)
