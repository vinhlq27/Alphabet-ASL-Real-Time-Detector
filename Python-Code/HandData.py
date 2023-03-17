#pip install opencv-python
#pip install mediapipe

import cv2 as cv
import mediapipe as mp
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=1, modelComplex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def extract_feature(self, input_image):
        mp_hands = self.mpHands
        mp_drawing = self.mpDraw
        #image = cv.imread(input_image)
        image = input_image
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            min_detection_confidence=0.1) as hands:
            while True:
                results = hands.process(cv.flip(cv.cvtColor(image, cv.COLOR_BGR2RGB), 1))
                image_height, image_width, _ = image.shape

                # Draw hand landmarks of each hand.
                if not results.multi_hand_landmarks:
                    # Here we will set whole landmarks into zero as no handpose detected
                    # in a picture wanted to extract.

                    # Wrist Hand
                    wristX = 0
                    wristY = 0
                    wristZ = 0

                    # Thumb Finger
                    thumb_CmcX = 0
                    thumb_CmcY = 0
                    thumb_CmcZ = 0

                    thumb_McpX = 0
                    thumb_McpY = 0
                    thumb_McpZ = 0

                    thumb_IpX = 0
                    thumb_IpY = 0
                    thumb_IpZ = 0

                    thumb_TipX = 0
                    thumb_TipY = 0
                    thumb_TipZ = 0

                    # Index Finger
                    index_McpX = 0
                    index_McpY = 0
                    index_McpZ = 0

                    index_PipX = 0
                    index_PipY = 0
                    index_PipZ = 0

                    index_DipX = 0
                    index_DipY = 0
                    index_DipZ = 0

                    index_TipX = 0
                    index_TipY = 0
                    index_TipZ = 0

                    # Middle Finger
                    middle_McpX = 0
                    middle_McpY = 0
                    middle_McpZ = 0

                    middle_PipX = 0
                    middle_PipY = 0
                    middle_PipZ = 0

                    middle_DipX = 0
                    middle_DipY = 0
                    middle_DipZ = 0

                    middle_TipX = 0
                    middle_TipY = 0
                    middle_TipZ = 0

                    # Ring Finger
                    ring_McpX = 0
                    ring_McpY = 0
                    ring_McpZ = 0

                    ring_PipX = 0
                    ring_PipY = 0
                    ring_PipZ = 0

                    ring_DipX = 0
                    ring_DipY = 0
                    ring_DipZ = 0

                    ring_TipX = 0
                    ring_TipY = 0
                    ring_TipZ = 0

                    # Pinky Finger
                    pinky_McpX = 0
                    pinky_McpY = 0
                    pinky_McpZ = 0

                    pinky_PipX = 0
                    pinky_PipY = 0
                    pinky_PipZ = 0

                    pinky_DipX = 0
                    pinky_DipY = 0
                    pinky_DipZ = 0

                    pinky_TipX = 0
                    pinky_TipY = 0
                    pinky_TipZ = 0

                    # Set image to Zero
                    annotated_image = 0

                    # Return Whole Landmark and Image
                    return (wristX, wristY, wristZ,
                            thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                            thumb_McpX, thumb_McpY, thumb_McpZ,
                            thumb_IpX, thumb_IpY, thumb_IpZ,
                            thumb_TipX, thumb_TipY, thumb_TipZ,
                            index_McpX, index_McpY, index_McpZ,
                            index_PipX, index_PipY, index_PipZ,
                            index_DipX, index_DipY, index_DipZ,
                            index_TipX, index_TipY, index_TipZ,
                            middle_McpX, middle_McpY, middle_McpZ,
                            middle_PipX, middle_PipY, middle_PipZ,
                            middle_DipX, middle_DipY, middle_DipZ,
                            middle_TipX, middle_TipY, middle_TipZ,
                            ring_McpX, ring_McpY, ring_McpZ,
                            ring_PipX, ring_PipY, ring_PipZ,
                            ring_DipX, ring_DipY, ring_DipZ,
                            ring_TipX, ring_TipY, ring_TipZ,
                            pinky_McpX, pinky_McpY, pinky_McpZ,
                            pinky_PipX, pinky_PipY, pinky_PipZ,
                            pinky_DipX, pinky_DipY, pinky_DipZ,
                            pinky_TipX, pinky_TipY, pinky_TipZ,
                            annotated_image)

                annotated_image = cv.flip(image.copy(), 1)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Wrist
                    wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                    wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                    wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                    # Thumb Finger
                    thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
                    thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                    thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                    thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
                    thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                    thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                    thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
                    thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                    thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                    thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                    thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                    thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                    # Index Finger
                    index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
                    index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                    index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                    index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
                    index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                    index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                    index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
                    index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                    index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                    index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                    index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                    index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                    # Middle Finger
                    middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                    middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                    middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                    middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                    middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                    middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                    middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
                    middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                    middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                    middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                    middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                    middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                    # Ring Finger
                    ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
                    ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                    ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                    ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
                    ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                    ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                    ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
                    ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                    ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                    ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
                    ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                    ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                    # Pinky Finger
                    pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
                    pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                    pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                    pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
                    pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                    pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                    pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
                    pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                    pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                    pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
                    pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                    pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

                    # Draw the hand
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS)

                return (wristX, wristY, wristZ,
                        thumb_CmcX, thumb_CmcY, thumb_CmcZ,
                        thumb_McpX, thumb_McpY, thumb_McpZ,
                        thumb_IpX, thumb_IpY, thumb_IpZ,
                        thumb_TipX, thumb_TipY, thumb_TipZ,
                        index_McpX, index_McpY, index_McpZ,
                        index_PipX, index_PipY, index_PipZ,
                        index_DipX, index_DipY, index_DipZ,
                        index_TipX, index_TipY, index_TipZ,
                        middle_McpX, middle_McpY, middle_McpZ,
                        middle_PipX, middle_PipY, middle_PipZ,
                        middle_DipX, middle_DipY, middle_DipZ,
                        middle_TipX, middle_TipY, middle_TipZ,
                        ring_McpX, ring_McpY, ring_McpZ,
                        ring_PipX, ring_PipY, ring_PipZ,
                        ring_DipX, ring_DipY, ring_DipZ,
                        ring_TipX, ring_TipY, ring_TipZ,
                        pinky_McpX, pinky_McpY, pinky_McpZ,
                        pinky_PipX, pinky_PipY, pinky_PipZ,
                        pinky_DipX, pinky_DipY, pinky_DipZ,
                        pinky_TipX, pinky_TipY, pinky_TipZ,
                        annotated_image)

    def extractData(self, path_to_img):
        (wristX, wristY, wristZ,
         thumb_CmcX, thumb_CmcY, thumb_CmcZ,
         thumb_McpX, thumb_McpY, thumb_McpZ,
         thumb_IpX, thumb_IpY, thumb_IpZ,
         thumb_TipX, thumb_TipY, thumb_TipZ,
         index_McpX, index_McpY, index_McpZ,
         index_PipX, index_PipY, index_PipZ,
         index_DipX, index_DipY, index_DipZ,
         index_TipX, index_TipY, index_TipZ,
         middle_McpX, middle_McpY, middle_McpZ,
         middle_PipX, middle_PipY, middle_PipZ,
         middle_DipX, middle_DipY, middle_DipZ,
         middle_TipX, middle_TipY, middle_TipZ,
         ring_McpX, ring_McpY, ring_McpZ,
         ring_PipX, ring_PipY, ring_PipZ,
         ring_DipX, ring_DipY, ring_DipZ,
         ring_TipX, ring_TipY, ring_TipZ,
         pinky_McpX, pinky_McpY, pinky_McpZ,
         pinky_PipX, pinky_PipY, pinky_PipZ,
         pinky_DipX, pinky_DipY, pinky_DipZ,
         pinky_TipX, pinky_TipY, pinky_TipZ,
         output_IMG) = self.extract_feature(path_to_img)

        # Shape the image features into a 1x3 array
        input_img = np.array([[[wristX], [wristY], [wristZ],
                               [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                               [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                               [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                               [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                               [index_McpX], [index_McpY], [index_McpZ],
                               [index_PipX], [index_PipY], [index_PipZ],
                               [index_DipX], [index_DipY], [index_DipZ],
                               [index_TipX], [index_TipY], [index_TipZ],
                               [middle_McpX], [middle_McpY], [middle_McpZ],
                               [middle_PipX], [middle_PipY], [middle_PipZ],
                               [middle_DipX], [middle_DipY], [middle_DipZ],
                               [middle_TipX], [middle_TipY], [middle_TipZ],
                               [ring_McpX], [ring_McpY], [ring_McpZ],
                               [ring_PipX], [ring_PipY], [ring_PipZ],
                               [ring_DipX], [ring_DipY], [ring_DipZ],
                               [ring_TipX], [ring_TipY], [ring_TipZ],
                               [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                               [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                               [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                               [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])
        return input_img


if __name__ == "__main__":
    handDetector()