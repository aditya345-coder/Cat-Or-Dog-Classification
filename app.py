import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, TruePositives, FalsePositives, TrueNegatives,FalseNegatives, Precision, Recall, AUC 
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
from pathlib import Path
import os
import cv2


# -------Title-------
st.title("Cat Or Dog Classifier")


# -------------- File Uploader ------------
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    st.write("""Preview 👀 Of Given Image!""")
    
    if image is not None:
        st.image(image, use_column_width=True)
    
    st.write("""Now, you are just one step ahead of prediction.""")
    st.write("""**Just Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! 😄**""")
except:
    st.write("""### ❗ Any Picture hasn't selected yet!!!""")


# ---------------- Predict Button -------------
st.text("""""")
submit = st.button("👉🏼 Predict")


# ------------------------ Model ----------------
def processing(testing_image_path):
    IMG_SIZE = 224
    img = load_img(testing_image_path, 
            target_size=(IMG_SIZE, IMG_SIZE))
     
    input_data = tf.constant(img)  # Your image data with shape (224, 224, 3)
    input_data = tf.expand_dims(img, 0)
    try:
        prediction = loaded_model.predict(np.array(input_data))
    except Exception as e:
        print(e)
    print(prediction)
    return prediction

def generate_result(prediction):
    st.write("""*****RESULT*****""")
    if prediction[0][0] < 0.5:
        st.write("""Model predicts it as an image of a Dog""")
    else:
        st.write("""Model predicts it as an image of a Cat""")
        
        
# ----------------------- Predict Button Clicked ------------------
if submit:
    try: 
        save_img("temp_dir/test_image.jpg", img_array)
        image_path = "temp_dir/test_image.jpg"
        
        st.write("Predicting.....")
        
        model_path_h5 = "models/model.h5"
        model_path_json = "models/model.json"
        
        json_file = open(model_path_json,'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_path_h5)
        
        metrics = [TruePositives(name='tp'), 
                   FalsePositives(name='fp'), 
                   TrueNegatives(name='tn'), 
                   FalseNegatives(name='fn'),
                   BinaryAccuracy(name='accuracy'), 
                   Precision(name='precision'), 
                   Recall(name='recall'), 
                   AUC(name='auc')]
        
        loaded_model.compile(optimizer = Adam(learning_rate = 0.001),
                             loss = "binary_crossentropy",metrics = metrics)
        prediction = processing(image_path)
        generate_result(prediction)

    except:
        st.write("!Oops... Something went wrong")
        
		