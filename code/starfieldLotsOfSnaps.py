import pyrealsense2 as rs
import cv2
import numpy as np
# Configure color stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
img_counter = 0
frame_counter = 0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        frame_counter += 1
        color_image = np.asanyarray(color_frame.get_data())
        frame = color_image
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif frame_counter % 10 == 0:
            # every 10 frames
            img_name = "starfield_frame_{}.png".format(frame_counter//10)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        # Press esc or 'q' to close the image window
        if k & 0xFF == ord('q') or k == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()
