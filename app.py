import streamlit as st
import base64
import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.cm as cm
from PIL import Image

def homepage():
    st.write("""
    ## Welcome to the Malaria Cell Detection WebApp
    """)

    st.markdown("<p>    Malaria is a life-threatening disease caused by parasites that are transmitted to people "
                "through the bites of infected female Anopheles mosquitoes. It is preventable and curable. </p> <ol>"
                "<ol>"
                "   <li>In 2017, there were an estimated 219 million cases of malaria in 90 countries.</li>"
                "   <li>Malaria deaths reached 435 000 in 2017.</li>"
                "   <li>The WHO African Region carries a disproportionately high share of the global malaria burden. In "
                "2017, the region was home to 92% of malaria cases and 93% of malaria deaths.</li> "
                "</ol>"
                "<p>Malaria is caused by Plasmodium parasites. The parasites are spread to people through the bites "
                "of infected female Anopheles mosquitoes, called \"malaria vectors.\" There are 5 parasite species "
                "that cause malaria in humans, and 2 of these species – P. falciparum and P. vivax – pose the "
                "greatest threat.</p>", unsafe_allow_html=True)


def upload():
    st.write("""
    ## Upload and Detect
    """)
    file = st.file_uploader('Choose a image file', type='png')
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels='BGR')
        st.write('You selected `%s`' % file.name)
        model = tf.keras.models.load_model(r'models\basic_malaria_pos_neg_v1.h5')
        pred, heatmap = apply_gradcam(model, opencv_image)
        st.write('Prediction: ' + pred)
        super_imposed = display_gradcam(opencv_image, heatmap)
        st.image(super_imposed)



def video():
    st.write("""
    ## Demo Video
    """)
    st.video(r'demo.mp4')


def report():
    st.write("""
    ## Report
    """)
    report_dir = r'report.pdf'
    st_pdf_display(report_dir)

def file_selector(folder_path='.'):
    file = st.file_uploader('Choose a image file', type='png')

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels='BGR')
    return file

def st_pdf_display(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title('Malaria Cell Detection')

    PAGES = {
        'Home': homepage,
        'Upload and Classify': upload,
        'Demo Video': video,
        'Report': report}
    st.sidebar.title('Navigation')
    PAGES[st.sidebar.radio('Go To', ('Home', 'Upload and Classify', 'Demo Video', 'Report'))]()
    # st.write(option_chosen)

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

    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)  # this converts it into RGB
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 3)

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
    img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = Image.fromarray(img)
    img = keras.preprocessing.image.img_to_array(img_path)

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
    main()