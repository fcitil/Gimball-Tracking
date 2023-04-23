import time 
import serial

def start_link(port:str = "/dev/tty1"):
    ser = serial.Serial(port=port, baudrate= 9600)  # replace 'COM3' with the serial port of your Arduino
    return ser

def send_data(ser, pan:int,tilt:int):
    try:
        ser.write((str(max(min(pan,2500),800))+ "," + str(max(min(tilt,2500),800)) +"\n").encode())
        

    except:
        print("error occured in sending data")
    time.sleep(0.02)