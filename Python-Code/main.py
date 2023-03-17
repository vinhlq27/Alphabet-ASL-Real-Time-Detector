#!pip install keras
#!pip install numpy
#!pip install opencv-python

from keras.models import load_model
import numpy as np
import cv2
import HandData as HD

# Load the model
model = load_model('model_ASL_shuffle.h5')      # The model path

# Encoding the prediction
classes = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'K': 9,
    'L': 10,
    'M': 11,
    'N': 12,
    'O': 13,
    'P': 14,
    'Q': 15,
    'R': 16,
    'S': 17,
    'T': 18,
    'U': 19,
    'V': 20,
    'W': 21,
    'X': 22,
    'Y': 23
}

'''# Load the testing image
path_to_image = "A (4).jpg"
input_img = detector.extractData(path_to_image)'''

# Initialize webcam
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
detector = HD.handDetector()

while True:
    success, img = cap.read()
    input_img = detector.extractData(img)
    #img = detector.findHands(img)
    # Print prediction with defined classes
    predict = model.predict(input_img)
    predict_classes = np.argmax(predict, axis=1)
    for alphabet, value in classes.items():
        if value == predict_classes[0]:
            print("The possible letter:", alphabet)
            # Display
            cv2.putText(img, f'Letter : {str(alphabet)}', (300, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 255, 0), 3)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break