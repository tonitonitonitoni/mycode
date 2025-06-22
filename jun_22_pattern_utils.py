import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN

def identify_cluster(pat):
    pat_cols = [c for x, y, c in pat]
    colors = sorted(pat_cols)
    if is_collinear(pat):
        if colors == [0, 1, 1]:  # Green, Blue, Blue
            cluster_index = 1
        elif colors == [0, 0, 0]:  # Green, Green, Green
            cluster_index = 3
        else:
            print("Can't identify cluster")
            cluster_index = False
    else:
        if colors == [0, 0, 1]: #Green, Green, Blue
            cluster_index = 0
        elif colors == [1, 1, 1]: #Blue, Blue, Blue
            cluster_index = 2
        else:
            print("Can't identify cluster")
            cluster_index = False

    return cluster_index

jetson = 1
if jetson:
    def show_image(image, axis=True, show=False):
        cv2.imshow("image", image)
        # Press esc or 'q' to close the image windows
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
else:
    def show_image(image, axis=True, show=False):
        # cv2.imshow doesn't work on Mac
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not axis:
            plt.axis("off")
        if show:
            plt.show()

## Processing
def find_centroids(img, visualize=False, vis_masks=False, area_threshold=10, dist_threshold=5, blue_threshold=120):
    # HSV ranges for green and blue
    lower_green = np.array([40, 100, 40])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([100, 100, 40])
    upper_blue = np.array([140, 255, 255])

    # convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Extract blue channel
    blue = img[:, :, 0]
    _, blue_blue_mask = cv2.threshold(blue, blue_threshold, 255, cv2.THRESH_BINARY)

    # combine blue masks for the ULTIMATE BLUE
    blue_mask = cv2.bitwise_and(blue_mask, blue_blue_mask)

    led_centroids = []
    i = 0
    for mask in [green_mask, blue_mask]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get LED centroids
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > area_threshold:  # Area threshold
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    new_centroid = (cx, cy)
                    check_dist = [distance(new_centroid, c) for c in led_centroids]
                    if len(check_dist) > 0:
                        if min(check_dist) > dist_threshold:  # Distance threshold
                            led_centroids.append((cx, cy, i))
                    else:
                        led_centroids.append((cx, cy, i))
        i += 1
    led_centroids = np.array(led_centroids)
    if len(led_centroids)==0:
        print("No LEDs detected.")
        return False
    else:
        return led_centroids

# Patterning
def find_patterns(points):
    centroids = points[:, 0:2]
    clustering = DBSCAN(eps=125, min_samples=3).fit(centroids)
    labels = clustering.labels_
    # Extract clusters
    num = len(set(labels)) - (1 if -1 in labels else 0)
    pats = []
    for j in range(num):
        pattern = points[labels == j]
        if len(pattern) == 3:
            pats.append(pattern)
    return pats, num

# Patterning
def visualize_patterns(img, pats, num, fid=None):
    output = img.copy()
    cmap = plt.get_cmap('YlOrRd', num)

    show_image(output)
    if len(pats) != 0:
        for j, pattern in enumerate(pats):
            pts = pattern[:, 0:2].astype(np.float32)
            pat_centre = np.mean(pts, axis=0)
            pt_distances = [distance(p, pat_centre) for p in pts]
            radius = int(np.ceil(max(pt_distances)))+10
            idx = identify_cluster(pattern)
            cv2.circle(output, (int(pat_centre[0]), int(pat_centre[1])), radius, (255, 255, 255), 1)
            cv2.putText(output, f'{idx}', (int(pat_centre[0])+radius, int(pat_centre[1])+radius), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

            c = cmap(j)
            xs = pattern[:, 0]
            ys = pattern[:, 1]
            plt.scatter(xs, ys, s=10, color=c, label=f'Cluster {idx}')
        plt.legend()
        if fid:
            plt.title(f"Detected Patterns in image {fid}")
        else:
            plt.title("Detected Patterns: Map")
        plt.axis("off")
        plt.show()
    else:
        if fid:
            plt.title(f"No patterns found for image {fid} .")
        else:
            plt.title("No patterns found for image.")
        plt.show()

# Identify pattern
def is_collinear(points, tol=2.5):
    # The AI wrote this.
    # Use PCA to determine if 4 points lie on a line.
    pts = np.array(points)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    _, s, _ = np.linalg.svd(centered)
    return s[1] < tol  # second singular value close to 0

# Useful math functions
def make_vector(p1, p2):
    p1_coords = np.array(p1[0:2])
    p2_coords = np.array(p2[0:2])
    return p2_coords - p1_coords

def distance(p1, p2):
    return np.linalg.norm(make_vector(p1, p2))

def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(abs(cos_angle))
    return int(np.round(np.rad2deg(angle), 0))

# Identify pattern
def angles_in_triangle(A, B, C):
    #Sort triangle vertices by ascending angle
    side_AB = make_vector(B, A)
    side_AC = make_vector(C, A)
    side_BC = make_vector(C, B)
    angle_B = angle_between(side_AB, side_BC)
    angle_A = angle_between(side_AB, side_AC)
    angle_C = angle_between(side_BC, side_AC)
    point_list = [[angle_A, A], [angle_B, B], [angle_C, C]]
    sorted_point_list = sorted(point_list)
    coords = [point[1:] for point in sorted_point_list]
    return coords

def arrange_by_distance(A, B, C):
    AB = distance(A, B)
    AC = distance(A, C)
    BC = distance(B, C)
    # lines arranged  *  *     *
    #A B C; A C B ; B A C; B C A; C A B ; C B A
    if AB < BC < AC:
        return A, B, C
    if AC < BC < AB:
        return A, C, B
    if AB < AC < BC:
        return B, A, C
    if BC < AC  < AB:
        return B, C, A
    if AC < AB  < BC:
        return C, A, B
    if BC < AB < AC:
        return C, B, A

def relative_transformation_with_scale(A, B):
    """
    An AI wrote this.

    Calculates the best-fit similarity transform (rotation, translation, scale)
    that maps points A to points B using a scaled version of the Kabsch algorithm.

    Args:
        A: NxD numpy array of source points
        B: NxD numpy array of destination points

    Returns:
        R: DxD rotation matrix
        t: Dx1 translation vector
        s: scalar scale factor
    """
    assert A.shape == B.shape, "Point sets must have the same shape"

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA.T @ BB

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    var_A = np.sum(AA ** 2)
    s = np.sum(S) / var_A

    # Compute translation
    t = centroid_B - s * R @ centroid_A

    return R, t, s

def apply_similarity_transform(A, R, t, s):
    # An AI wrote this.
    # A = input point set
    # R, t, s are calculated by the relative_transform_with_scale function

    return (s * (R @ A.T)).T + t



def sort_cluster(pat):
    if is_collinear(pat):
        sorted_points = arrange_by_distance(*pat)
    else:
        sorted_points = angles_in_triangle(*pat)
    return sorted_points


def points_to_array(point_list):
    point_array = []
    for points in point_list:
        for point in points:
            point_array.append(point)
    unnested_list = []
    for point in point_array:
        if len(point) < 2:
            unnested_list.append(point[0])
        else:
            unnested_list.append(point)
    return np.array(unnested_list)


def pose_from_sample(sample_obs, cluster_locs):
    sample_centroids = find_centroids(sample_obs, area_threshold=10, dist_threshold=5)
    if len(sample_centroids)>0:
        patterns, num_clusters = find_patterns(sample_centroids)

        #if num_clusters>0:
        observed_point_list = []
        true_point_list = []
        for pat in patterns:
            cluster_index = identify_cluster(pat)
            if cluster_index:
                observed_points = pat[:, 0:2].tolist()
                sorted_observed_points = sort_cluster(observed_points)

                true_points = cluster_locs[str(cluster_index)]
                sorted_true_points = sort_cluster(true_points)
                observed_point_list.append(sorted_observed_points)
                true_point_list.append(sorted_true_points)

        opl = points_to_array(observed_point_list)
        tpl = points_to_array(true_point_list)
        if len(tpl)>0:
            R, t, s = relative_transformation_with_scale(opl, tpl)
            computed_points = apply_similarity_transform(opl, R, t, s)
            error = computed_points - tpl
            if np.linalg.norm(error) > 10:
                print("Large error: ", np.linalg.norm(error))
            sample_centre = np.array([int(sample_obs.shape[1] / 2), int(sample_obs.shape[0] / 2)])
            sample_vec = np.array([0, -30])
            position = apply_similarity_transform(sample_centre, R, t, s)
            angle = np.rad2deg(np.arctan2(R[1][0], R[0][0]))
            if angle < 0: angle += 360
            pose_dy = R @ sample_vec #.T @ R#np.dot(sample_vec, R)#sample_vec.T @ R
            result = {'position': position, 'angle': angle, 'gradient': pose_dy}
            return result


        else:
            return False
    else:
        return False
