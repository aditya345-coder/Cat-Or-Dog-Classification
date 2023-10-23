
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
from pathlib import Path
import os
import random
import webbrowser
import cv2

# ------------------------ Model ----------------
def processing(testing_image_path):
    # IMG_SIZE = 224
    # img = load_img(testing_image_path, 
    #         target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")   
    
    
    pass
    # return prediction

def generate_result(prediction):
    pass
        
image_path = "temp_dir/test_image.jpg"
model_path_h5 = "model.h5"
model_path_json = "model.json"
json_file = open(model_path_json,'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path_h5)
loaded_model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')
img=image_path
im = tf.constant(img, dtype=tf.float32)
im = tf.expand_dims(im, axis = 0)
prediction = loaded_model.predict(img)   
if prediction[0] < 0.5:
    print("""Model predicts it as an image of a Cat""")
else:
    print("""Model predicts it as an image of a Dog""") 
# prediction = processing(image_path)
# generate_result(prediction)