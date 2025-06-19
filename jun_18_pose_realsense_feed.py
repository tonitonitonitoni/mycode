import pyrealsense2 as rs
from jun_18_pattern_utils import *
import json

# Configure pipeline and enable color stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load cluster data
with open('cluster_data.json', 'r') as f:
    cluster_locs = json.load(f)

# Load map
star_map = cv2.imread('jun17_map.png')

try:
    while True:
        map_copy = star_map.copy()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())

        # Plot the robot's pose on the observed image for comparison.
        img_centre = np.array([int(image.shape[1] / 2), int(image.shape[0] / 2)])
        fwd_vec = np.array([0, 30])
        arrow_head = img_centre + fwd_vec
        cv2.arrowedLine(image, img_centre, arrow_head, (0, 0, 255), 2, tipLength = 0.25)

        # Find the pose from the image.
        position, angle, pose_dy = pose_from_sample(image, cluster_locs)

        # Plot the pose on the map.
        start_point = position
        end_point = np.array(start_point)+np.array(pose_dy)

        cv2.arrowedLine(map_copy, start_point, end_point, (0, 0, 255), 2, tipLength = 0.25)
        cv2.putText(map_copy, f'position: ({position[0]:.1f}, {position[1]:.1f}) angle:{angle:.1f}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #Show the images
        cv2.namedWindow('Observed Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Observed Image', image)

        cv2.namedWindow('Pose on Map', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Pose on Map', map_copy)

        # Press esc or 'q' to close the image windows
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()