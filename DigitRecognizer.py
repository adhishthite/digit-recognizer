''' DIGIT RECOGNIZER '''

# Imports
import cv2
from tensorflow.contrib import keras
import numpy as np

# Image Processing & Prediction
def recognize_digit():

    # Initialization of Variables
    frame_size = 240
    x, y, w, h = 0, 0, frame_size, frame_size

    # Keras Model Initialization
    model = keras.models.load_model('cnn-mnist-model.h5')

    # Initialize the Camera
    camera = cv2.VideoCapture(0)

    while True:

        # Grab the image
        return_value, image = camera.read()

        # Dummy Variable
        original_image = image

        # Highlight the readable area
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Image Processing
        image = cv2.resize(image[y:y + h, x:x + w], (28, 28))   # Downsize to 28x28 pixels
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # Convert to GRAYSCALE
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Sharpen the image

        # Image ready for prediction
        x_pred = im_bw.reshape(28, 28, 1)   # Reshape to 28x28x1 (size*size*color_channel)
        batch = np.array([x_pred])  # Create a batch (batch_size * size * size * color_channel)

        output = str(model.predict_classes(batch, verbose=0))

        # Add the ouput to the visible area of the camera
        cv2.putText(original_image, "Predicted Value is " + output, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(original_image, "Press \'E\' to Quit", (10, 340), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow('image', original_image)

        # Press 'E' to quit
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


def main():
    recognize_digit()

if __name__ == '__main__':
    main()