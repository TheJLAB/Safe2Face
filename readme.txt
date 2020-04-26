Safe2Face project

Safe2Face.py --> file to be loaded in a raspberryPI equipped with a sensor hat.
This sw allows gesture records on a file stored on a RPI. 

TrainCNN.py  --> ML file
preProcess1.py  --> ML file

How it works : 
  launch the SW by python safe2face.py
  Wait for the S (standby) to be displayed on the led matrix
  press Joystick Up : "I"  is displayed (Init) then "R" (recording)
  record your gesture.
  press Joystick Down :Record is stopped & "S" is displayed (Stand by)
