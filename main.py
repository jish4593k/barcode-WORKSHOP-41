import cv2
from pyzbar.pyzbar import decode  # Pyzbar is a library for barcode and QR code decoding
import numpy as np
import keras
from keras.models import load_model
import argparse


def detect_and_recognize_barcode(image_path, model_path):
    # Load the image
    image = cv2.imread(image_path)

    # Decode barcodes and QR codes in the image
    decoded_objects = decode(image)

   
    model = load_model(model_path)

    for obj in decoded_objects:
        # Extract the barcode data
        barcode_data = obj.data.decode('utf-8')
        print(f"Detected Barcode Data: {barcode_data}")

        # Preprocess the barcode data for recognition (you may need to customize this)
        preprocessed_data = preprocess_data(barcode_data)

        # Use the Keras model for content recognition
        content = model.predict(preprocessed_data)
        print(f"Recognized Content: {content}")

        # Draw a bounding box around the detected barcode
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array(points))
        else:
            hull = points
        cv2.polylines(image, [hull], True, (0, 255, 0), 3)

    # Display the image with bounding boxes
    cv2.imshow("Detected Barcodes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Preprocessing function (customize this according to your model's input requirements)
def preprocess_data(data):
    # Example: Convert string to integer and normalize
    data = int(data) / 255.0
    return np.array(data).reshape(1, -1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    parser.add_argument("-m", "--model", required=True, help="Path to the Keras model for content recognition")
    args = parser.parse_args()

    detect_and_recognize_barcode(args.image, args.model)
