import time 
import serial

def start_link(port:str = "/dev/tty1"):
    ser = serial.Serial(port=port, baudrate= 9600)  # replace 'COM3' with the serial port of your Arduino
    return ser

def send_data(ser, pan:int,tilt:int):
    try:
        angle = max(min(pan, 135), 45)
        ser.write((str(angle)+"\n").encode())
        print("pan angle set to", angle)
    
        angle = max(min(tilt, 180), 0)
        ser.write((str(angle)+"\n").encode())
        print("tilt angle set to", angle)
        

    except:
        print("error occured in sending data")
    time.sleep(0.1)