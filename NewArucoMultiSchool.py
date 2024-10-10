import pyrealsense2 as rs
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

# Side length of the ArUco marker in meters
markerSize = 70.48
# Calibration parameters yaml file
calFile = 'RSCalChessboard.yaml'
cv_file = cv2.FileStorage(calFile, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()
dist = cv_file.getNode('D').mat()
cv_file.release()

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
parameters = cv2.aruco.DetectorParameters_create()

boardVert = 112.2
boardHoriz = 140.8 # millimetres
markerPositions={25:(markerSize/2, markerSize/2, 0), 26:(markerSize/2, markerSize/2 + boardVert, 0), 20:(markerSize/2 + boardHoriz, markerSize/2, 0), 27:(markerSize/2 + boardHoriz, markerSize/2 + boardVert, 0)} 

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


# Configure color stream
pipeline = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

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
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Check that at least one ArUco marker was detected
        if ids is not None:
            total_poseX = []
            total_poseY = []
            total_yaw = []
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, markerSize, mtx, dist)
            for i, id in enumerate(ids):
                # Store the translation (i.e. position) information
                rvec = rvecs[i]
                tvec = tvecs[i]
                rvec_flip = -1*rvec
                tvec_flip = -1*tvec[0]
                Rmat, J = cv2.Rodrigues(rvec_flip)

                t_RW = np.dot(Rmat, tvec_flip)


                # Store the rotation information
                r = R.from_matrix(Rmat[0:3, 0:3])
                quat = r.as_quat()

                # Quaternion format
                transform_rotation_x = quat[0]
                transform_rotation_y = quat[1]
                transform_rotation_z = quat[2]
                transform_rotation_w = quat[3]

                # Euler angle format in radians
                roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                               transform_rotation_y,
                                                               transform_rotation_z,
                                                               transform_rotation_w)

                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)

#                print(Rmat)
#                print(tvec_flip)
                marker_posn = markerPositions[id[0]]
                marker_x = float(marker_posn[0])
                marker_y = float(marker_posn[1])
                marker_str = f'Marker{id}, x: {marker_x:.1f}, y: {marker_y:.1f}'
                marker_textlocn=corners[i][0][1]
                m_loc=int(marker_textlocn[0]), int(marker_textlocn[1])
                #cv2.putText(color_image, marker_str, m_loc, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

                tvec_str = f'tvec x:{t_RW[0]:.1f} y:{t_RW[1]:.1f} yaw:{yaw_z:.1f}'
                pos = corners[i][0][0]
                t_loc = (int(pos[0]), int(pos[1])-20)
                #cv2.putText(color_image, tvec_str, t_loc, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

                poseX=marker_x+t_RW[0]
                poseY=marker_y+t_RW[1]
                total_poseX.append(poseX)
                total_poseY.append(poseY)
                total_yaw.append(yaw_z)
                pose_textloc = corners[i][0][2]
                p_loc = (int(pose_textloc[0]), int(pose_textloc[1]))
                p_str = f' pose x:{poseX:.1f}, y:{poseY:.1f}, yaw:{yaw_z:.1f}'
                cv2.putText(color_image, p_str, p_loc, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                # Draw the axes on the marker
                cv2.drawFrameAxes(color_image, mtx, dist, rvecs[i], tvecs[i], 50)
            pX=np.mean(total_poseX)
            pY=np.mean(total_poseY)
            pYaw=np.mean(total_yaw)
            avP_str=f'X: {pX:.0f} Y: {pY:.0f} Yaw: {pYaw:.0f}'
            centre=(gray.shape[1]/2, gray.shape[0]/2)
            cv2.putText(color_image, avP_str, (320, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            print(f'X: {pX:.1f}, Y: {pY:.1f}, Yaw: {pYaw:.1f}')
            # print(centre)

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
