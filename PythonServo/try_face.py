from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import cv2
import os

import time
from threading import Thread
from os.path import realpath, dirname, join
import serial
from libs.objcenter import ObjCenter
from libs.pid import PID
from libs.servocontrol import start_link, send_data


objX = 0
objY = 0
centerX = 0
centerY = 0


def obj_center():

    vs = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    time.sleep(2.0)

    obj = ObjCenter(join(realpath(dirname(__file__)), "haar.xml"))

    global objX
    global objY
    global centerX
    global centerY
    global width
    global height

    ret, img = vs.read()
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    while True:
        # grab the current frame
        ret, frame = vs.read()
        #image resizing
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        #mirror the frame
        frame = cv2.flip(frame, 1)

        # calculate the center of the frame as this is where we will
        # try to keep the object
        (H, W) = frame.shape[:2]
        centerX = W // 2
        centerY = H // 2
        cv2.circle(frame, (centerX, centerY), 5, (0, 0, 255), -1)

        # find the object's location
        objectLoc = obj.update(frame)
        try:
            ((objX, objY), rect) = objectLoc
            cv2.circle(frame, (objX, objY), 5, (0, 255, 0), -1)

            # extract the bounding box and draw it
            if rect is not None:
                (x, y, w, h) = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        except:
            pass
        cv2.imshow("Selam Furkan", frame)
        cv2.waitKey(1)
        


def pid_processX(p, i, d):

    global outputX

    p = PID(p, i, d)
    p.initialize()

    while True:
        error = centerX - objX
        # print(centerCoord,objCoord,error)
        outputX = p.update(error)
        time.sleep(0.1)


def pid_processY(p, i, d):

    global outputY
    p = PID(p, i, d)
    p.initialize()

    while True:
        error = centerY - objY
        # print(centerCoord,objCoord,error)
        outputY = p.update(error)
        time.sleep(0.1)


def send_angl(ser):
    #start the serial port
    #time.sleep(5)
    while True:
        try:
            pan = int(outputY+90)
            tilt = int(outputX)
            send_data(ser,pan,tilt)
        except:
            pass
        
        
def plotter():
    time.sleep(10)
    x_data, y_data ,time_data = [], [] , []

    figure = plt.figure()
    linex, = plt.plot_date(time_data, x_data, '-')
    liney, = plt.plot_date(time_data, y_data, '-')

    def update(frame):
        x_data.append(outputX)
        y_data.append(outputY)
        time_data.append(datetime.now())

        liney.set_data(time_data, y_data)
        linex.set_data(time_data, x_data)
        figure.gca().relim()
        figure.gca().autoscale_view()
        return linex, liney


    anim = FuncAnimation(figure, update, interval=200 , save_count=50)
    plt.show()
    

# TODO : can uset the multiprocessing instead of threading
detection_thread = Thread(target=obj_center)
detection_thread.daemon = True
detection_thread.start()
time.sleep(5)
pidy_thread = Thread(target=pid_processY, args=(0.09, 0.08, 0.002))
pidx_thread = Thread(target=pid_processX, args=(0.11, 0.10, 0.002))

ser = start_link("COM7")


command_thread =  Thread(target=send_angl, args = [ser])

command_thread.daemon=True
pidx_thread.daemon = True
pidy_thread.daemon = True

pidy_thread.start()
pidx_thread.start()

command_thread.start()

#plotter_thread = Thread(target=plotter)
#plotter_thread.daemon = True
#plotter_thread.start()

#plotter()

while True:
    time.sleep(1)
    try:
        print(f"object position x:{objX} y:{objY}")
        # print("center position x: , y: " ,centerX, centerY)
        print(f"output x:{outputX} y:{outputY}")
    except:
        ser.close()
        pass
    pass