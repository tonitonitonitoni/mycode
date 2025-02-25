This is my code, mostly for personal use but I want to explain it better.
# My Device-Agnostic To Do list
https://github.com/oz182/3D-Navigation---DWA-and-A-

## old stuff
- Upload important files from Jetson PycharmProjects
- https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/export_ply_example.py
- examine pointcloud in rviz with roslaunch realsense2_camera rs_camera.launch filters:=pointcloud
- 3D reconstruction with Open3D's realsense_recorder.py https://www.open3d.org/docs/release/tutorial/reconstruction_system/capture_your_own_dataset.html
- and 3D mobile robots code, https://github.com/luigifreda/3dmr/tree/main  For NBV Planner ported to Noetic see here https://github.com/luigifreda/3dmr/blob/main/README.exploration.md#3d-exploration-with-drones
- Try Voxfield (this at least is ready for Noetic) https://github.com/VIS4ROB-lab/voxfield
- https://github.com/ntnu-arl/rhem_planner Uncertainty-aware Receding Horizon Exploration and Mapping didn't work
- https://github.com/robotic-esp/see-public Surface Edge Explorer
- https://simulation.readthedocs.io/en/latest/ OnOrbit ROS this one kind of works but the satellite doesn't move

## later
- https://dev.intelrealsense.com/docs/tensorflow-with-intel-realsense-cameras
- set up the Arduino IDE on the Jetson, install rosserial and try to dim an LED on the arduino leonardo, *with ROS!* 
- then see if you can PWM some motors that way OR via pyserial
- https://automaticaddison.com/how-to-control-a-robots-velocity-remotely-using-ros/
- https://automaticaddison.com/how-to-publish-wheel-odometry-information-over-ros/
- https://www.youtube.com/watch?v=M8BlIjaz7pU dead reckoning with arduino and ROS noetic
- Re-do the tutorials from jetson-inference
- https://github.com/IntelRealSense/realsense-ros/wiki/SLAM-with-D435i

# Programs in this repo
## ArucoRWMulti.  
This code finds a real-world pose from multiple Aruco Markers.  It is a work in progress. It is mostly taken from several blogs.  First, Automatic Addison helped me with the rotation matrix code.  
The biggest new development is the transformation from camera coords into real world coords.  This was provided by OutOfTheBOTS on Youtube.  I will put links in here later.
It is currently written in OpenCV 4.10. 

## ArucoRealSense.  
This is simple marker detection code, but adapted from the classic OpenCV cap.read() format for the RealSense camera, which has a separate initialization procedure.
So far I have only been able to analyze the flat images from the 2D camera included with the RealSense, 
I hope soon to learn more about semantic segmentation of point cloud data.

The RSCalChessboard file is just to calibrate the RealSense camera. I am trying to calibrate it using a Charuco board but it is not going very well for me.

