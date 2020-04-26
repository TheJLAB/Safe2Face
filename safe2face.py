"""

		Safe2Face.py

		Software designed to drive Accelero & Gyro from a RPi Sense Hat
		
"""

import time

from sense_hat import SenseHat
from datetime import datetime
sense = SenseHat()
sense.clear()

#operation mode

fullTimeRec = True; #--> when set to true, Rpi will record all the mouvements done during joy Up and Joy down 

#fullTimeRec = False; #--> when set to true, Rpi will :
#															- init the process (init file) when joy up
#															- record event when joy left/joy right
#															- close the file when joy down


def getPressureValue():
	return sense.get_pressure()

def getIMUValue():
  	sense.set_imu_config(False, True, True)  # gyro + accelero

	gyro = sense.get_gyroscope()
	#gyroTxt = ("p: {pitch}, r: {roll}, y: {yaw}".format(**gyro))           //human reading
	gyroTxt2 = ("{pitch}, {roll}, {yaw}".format(**gyro))
	
	accelero = sense.get_accelerometer()
	#acceleroTxt = ("p: {pitch}, r: {roll}, y: {yaw}".format(**accelero ))  //human reading
	acceleroTxt2 = ("{pitch}, {roll}, {yaw}".format(**accelero ))
	
	#imuTxt = gyroTxt +"|"+acceleroTxt+";" 
	imuTxt = gyroTxt2 +"|"+acceleroTxt2+";" 
	return imuTxt 
	
	
	
	
# Main
sense.show_message("Standby")
sense.show_letter("S") #stand by

while True :
	for event in sense.stick.get_events():
		print(event.direction, event.action)

		if event.direction == "up":  #init sequence by pressing joy up
			sense.show_letter("I") #init
			time.sleep(1) # wait for the arm to be steady / prevent irrelevent data record during btn press/release gesture
			
			#file init
			curDateTime = time.strftime("%Y%m%d-%H%M%S")
			curTime = time.strftime("%H%M%S")
			fileTxt = open (curDateTime+'.txt' , "w")
			
			
			#header
			curTime = time.strftime("%H%M%S")
			fileTxt.write(curTime +"\r\n")
			pressure = sense.get_pressure()
			print(pressure)
			fileTxt.write("Stick Up : recorded data done w following parameters: Pstn : StandUp/ Arm : next to leg|\r\n")
			fileTxt.write("Pressure at  start: %4.2f" % (pressure) + "\r\n")
			looping= True
			#reading loop (until joy down)
			while looping: 
				for event in sense.stick.get_events():
					if fullTimeRec == False:
						accel=getIMUValue()
						accel = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]+"|"+accel
						fileTxt.write(accel)
						fileTxt.write("\n")
						print(accel)
					if event.direction == "down":
						looping = False
						
				if fullTimeRec == True:
						accel=getIMUValue()
						accel = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]+"|"+accel
						fileTxt.write(accel)
						fileTxt.write("\n")
						sense.show_letter("R") #recording
						print(accel)
			# end of loop
			
			pressure = sense.get_pressure()
			print(pressure)
			fileTxt.write("Pressure at End: %4.2f" % (pressure) + "\r\n")
			urTime = time.strftime("%H%M%S")
			fileTxt.write(curTime +"\r\n")
			fileTxt.close()
			sense.show_letter("S") #stand by
	


		