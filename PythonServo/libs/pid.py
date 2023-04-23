# import necessary packages
import time


class PID:
    def __init__(self, kP=1, kI=0, kD=0):
        # initialize gains
        self.kP = kP
        self.kI = kI
        self.kD = kD

    def initialize(self):
        # intialize the current and previous time
        self.currTime = time.time()
        self.prevTime = self.currTime
        self.result = 0
        # initialize the previous error
        self.prevError = 0

        # initialize the term result variables
        self.cP = 0
        self.cI = 0
        self.cD = 0

    def update(self, error, sleep=0.01,found=False):
        # pause for a bit
        self.prevTime = self.currTime
        time.sleep(sleep)

        # grab the current time and calculate delta time
        self.currTime = time.time()
        deltaTime = self.currTime - self.prevTime
        if found and self.result <= 700 and self.result >= -600:
            # delta error
            deltaError = error - self.prevError

            # proportional term
            self.cP = error

            # integral term
            self.cI += error * deltaTime

            # derivative term and prevent divide by zero
            self.cD = (deltaError / deltaTime) if deltaTime > 0 else 0

            # save previous time and error for the next update
            
            self.prevError = error
            self.result += sum([
            self.kP * self.cP,
            self.kI * self.cI,
            self.kD * self.cD])
        elif found and self.result >= 700:
            self.result = 700
        elif found and self.result <= -600:
            self.result = -600
        else:
            self.result = self.result * 0.99
            # sum the terms and return
        return self.result