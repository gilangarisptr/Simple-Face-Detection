import cv2
import argparse
from facedetection import FaceDetector


ap = argparse.ArgumentParser()
ap.add_argument("-f","--face", required = True, help = "path where the face cascade resides")
ap.add_argument("-i", "--image", required = True, help = "path where the image file")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Fd = FaceDetector(args["face"])
faceRects = Fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

print("I found {} face(s)".format(len(faceRects)))

for (x,y,w,h) in faceRects:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

#cv2.imshow("Faces", image)
cv2.imwrite("Faces result", image)
cv2.waitKey(0)