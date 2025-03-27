#model.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load model duy nhất 1 lần
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def enhance_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 1.5, 0, 255).astype(np.uint8)
    bright_img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    return bright_img

def get_embedding(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = enhance_brightness(img)
    img = cv2.resize(img, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features[0]