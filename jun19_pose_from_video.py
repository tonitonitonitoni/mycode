from jun_19_pattern_utils import *
import json
import os
cwd = '/home/toni/Workspace/jun19_files'

# Load cluster data
fname = os.path.join(cwd, 'cluster_data_jun19.json')
with open(fname, 'r') as f:
    cluster_locs = json.load(f)
#print(cluster_locs)
# Load map
fname = os.path.join(cwd, 'map_jun19.png')
star_map = cv2.imread(fname)

video_path = os.path.join(cwd, 'output4.avi') # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    map_copy = star_map.copy()
    ret, image = cap.read()

    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break  
    # Plot the robot's pose on the observed image for comparison.
    img_centre = np.array([int(image.shape[1] / 2), int(image.shape[0] / 2)])
    fwd_vec = np.array([0, -30])
    arrow_head = img_centre + fwd_vec
    cv2.arrowedLine(image, img_centre, arrow_head, (0, 0, 255), 2, tipLength=0.25)

    # Find the pose from the image.
    result = pose_from_sample(image, cluster_locs)

    if result:
        position = result['position']
        angle = result['angle']
        pose_dy = result['gradient']# Plot the pose on the map.
        start_point = int(position[0]), int(position[1])
        end_point = np.array(position) + np.array(pose_dy)
        end_point = int(end_point[0]), int(end_point[1])

        cv2.arrowedLine(map_copy, start_point, end_point, (0, 0, 255), 2, tipLength=0.25)
        cv2.putText(map_copy, f'position: ({position[0]:.1f}, {position[1]:.1f}) angle:{angle:.1f}', (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.namedWindow('Pose on Map', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Pose on Map', map_copy)
    else:
        cv2.putText(image, "No Pose Available", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # Show the images
    cv2.namedWindow('Observed Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Observed Image', image)

    key = cv2.waitKey(25)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()