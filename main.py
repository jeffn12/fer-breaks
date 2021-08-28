from fer import FER
import cv2
from operator import itemgetter
# import pprint

cam = cv2.VideoCapture(0)
detector = FER(mtcnn=True)
while True:
    ret_val, img = cam.read()
    emotions = detector.detect_emotions(img)
    if (emotions):
        angry, disgust, fear, happy, neutral, sad, surprise = itemgetter(
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")(emotions[0]["emotions"])
        print(angry, disgust, fear, happy, neutral, sad, surprise)
        if(angry > .5 or disgust > .5):
            print("take a break!")
        if(surprise > .6):
            print("chill out!")
            # chill out time
        # cv2.putText(img, pprint.pprint(
        #    emotions[0]["emotions"]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('my webcam', img)
    cv2.namedWindow('my webcam', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('my webcam', 600, 600)
    if cv2.waitKey(1) == 27:
        break  # press ESC to quit
cv2.destroyAllWindows()
