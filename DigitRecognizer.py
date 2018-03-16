import cv2
from tensorflow.contrib import keras
import numpy as np


def image_capture():

    camera = cv2.VideoCapture(0)

    frame_size = 240

    model = keras.models.load_model('cnn-mnist-model.h5')

    while True:
        x, y, w, h = 0, 0, frame_size, frame_size

        return_value, image = camera.read()

        original_image = image

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # cv2.imshow('FRAME', image)

        image = cv2.resize(image[y:y + h, x:x + w], (28, 28))

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        x_pred = im_bw.reshape(28, 28, 1)

        batch = np.array([x_pred])

        output = str(model.predict_classes(batch, verbose=0))

        cv2.putText(original_image, "Predicted Value is " + output, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(original_image, "Press \'S\' to Quit", (10, 340), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow('image', original_image)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            #cv2.imwrite('test.jpg', image)
            break
    camera.release()
    cv2.destroyAllWindows()


def image_process():

    im_gray = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite('test_proc.jpg', im_bw)

    x_pred = cv2.imread('test_proc.jpg', cv2.IMREAD_GRAYSCALE)

    predict_number(x_pred)

def predict_number(x_pred):

    model = keras.models.load_model('cnn-mnist-model.h5')

    x_pred = cv2.imread('test_proc.jpg', cv2.IMREAD_GRAYSCALE).reshape(28, 28, 1)

    batch = np.array([x_pred])

    print(model.predict_classes(batch, verbose=1))

    return str(model.predict_classes(batch, verbose=1))

def main():

    image_capture()

    #image_process()

if __name__ == '__main__':
    main()