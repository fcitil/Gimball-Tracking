# Import essential libraries
# import requests
import cv2
import numpy as np
# import imutils

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "https://192.168.34.141:8080/video"
img = cv2.VideoCapture(3)


# While loop to continuously fetching data from the Url
while True:
    
    ret,im = img.read()
    # print(im.shape)

    # img = imutils.resize(img, width=1080, height=1800)
    cv2.imshow('ehe', im)

    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
