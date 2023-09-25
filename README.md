# Brunel Autonomous Racing Work

https://github.com/DavenK84/Brunel-Autonomous/assets/145354427/af3c29d6-4dae-4a0a-9f3b-f86f39c84047
![image](https://github.com/DavenK84/Brunel-Autonomous/assets/145354427/cc02a7a0-ec4b-458c-88d5-2b91c07604c0)

The program "detect_ZED14.py" is an old iteration of the team's self driving algorithm. In this program, an RC car is driven around a track defined by two sets of blue and yellow cones. Attached to the car is a NVIDIA Jetson Nano board with a Stereolabs ZED2 stereo camera. the camera is used performs perception tasks, including object detection and object tracking, while simultaneously collecting pose estimation data with the camera's IMU sensor. Combining all this information, the car is able to generate a set of coordinates of the track. Using these coordinates, the vehicle controls team would be able to simulate a fastest lap on MATLAB Simulink, which included CAN signals for a car's wheel speed, wheel angle etc.




