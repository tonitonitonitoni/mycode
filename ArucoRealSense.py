import pyrealsense2 as rs
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

# Side length of the ArUco marker in meters
markerSize = 0.0528
# Calibration parameters yaml file
calFile = 'RSCalChessboard.yaml'
cv_file=cv2.FileStorage(calFile, cv2.FILE_STORAGE_READ)
mtx=cv_file.getNode('K').mat()
dist=cv_file.getNode('D').mat()
cv_file.release()
# Calibration parameters yaml file
camera_calibration_parameters_filename = 'RSCalChessboard.yaml'
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):

    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # color_colormap_dim = color_image.shape
        # Detect the markers
        corners, marker_ids, rejected = detector.detectMarkers(gray)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, marker_ids)
            rvecs, tvecs, obj_points = my_estimatePoseSingleMarkers(corners,
                                                                    markerSize,
                                                                    mtx,
                                                                    dist)
            for i, marker_id in enumerate(marker_ids):
                transform_translation_x = tvecs[i][0]
                transform_translation_y = tvecs[i][1]
                transform_translation_z = tvecs[i][2]

                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()

                transform_rotation_x = quat[0]
                transform_rotation_y = quat[1]
                transform_rotation_z = quat[2]
                transform_rotation_w = quat[3]

                roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                               transform_rotation_y,
                                                               transform_rotation_z,
                                                               transform_rotation_w)
                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)

                print(
                    f'translation x: {transform_translation_x}, translation y: {transform_translation_y}, translation z: {transform_translation_z}')
                print(f'roll: {roll_x}, pitch: {pitch_y}, yaw: {yaw_z}')

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)


        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()
