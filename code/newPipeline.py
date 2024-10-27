import pyrealsense2 as rs
import cv2
import numpy as np
from scaler_real_images import knn_predict, train_ideal_knn, scaler_from_real_images
from data_real_frame import data_real_frame

# train ideal KNN
excel_file='OctBData.pkl'
knn, colVals = train_ideal_knn(excel_file)

# get scaler from real images
realImageDir="realImgs"
realScaler=scaler_from_real_images(realImageDir, colVals)


# Configure color stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
img_counter = 0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        #get data from frame
        feature_vector, output = data_real_frame(color_image)
        pred = knn_predict(knn, realScaler, feature_vector, colVals)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        if k & 0xFF == ord('q') or k == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()