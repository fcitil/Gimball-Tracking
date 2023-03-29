#include <Arduino.h>
#include <Servo.h>
#include <SerialTransfer.h>
#include <Wire.h>

// Create a SerialTransfer object
SerialTransfer myTransfer;

// Create a Servo object
Servo Pan;
Servo Tilt;

struct __attribute__((packed)) STRUCT
{
  int pan;
  int tilt;
  bool cmd_attach;
  bool cmd_detach;
} packageStruct;

int pan_pin = 9;
int tilt_pin = 10;

int pan_max = 125;
int pan_min = 45;

int tilt_max = 180;
int tilt_min = 0;

bool attached = false;

void setup()
{
  // put your setup code here, to run once:
  Serial.begin(9600);

  myTransfer.begin(Serial);
  // Initialize the servo library
}

void loop()
{
  // put your main code here, to run repeatedly:
  if (myTransfer.available())
  {

    uint16_t recsize = 0;

    recsize = myTransfer.rxObj(packageStruct, recsize);

    Serial.print("Pan,tilt,attach,detach:");
    Serial.print(packageStruct.pan);
    Serial.print(",");
    Serial.print(packageStruct.tilt);
    Serial.print(",");
    Serial.print(packageStruct.cmd_attach);
    Serial.print(",");
    Serial.println(packageStruct.cmd_detach);

    if (packageStruct.cmd_attach && !attached)
    {
      Pan.attach(pan_pin);
      Tilt.attach(tilt_pin);
      attached = true;
      Serial.println("Attached");
    }
    else if (packageStruct.cmd_detach && attached)
    {
      Pan.detach();
      Tilt.detach();
      attached = false;
      Serial.println("Detached");
    }
    else if (attached
     && (packageStruct.pan <= pan_max && packageStruct.pan >= pan_min) 
     && (packageStruct.tilt <= tilt_max && packageStruct.tilt >= tilt_min))
    {
      Pan.write(packageStruct.pan);
      Tilt.write(packageStruct.tilt);
    }

  }
}
