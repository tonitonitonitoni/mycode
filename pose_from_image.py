from jun_18_pattern_utils import *
map = cv2.imread('jun17_map.png')
show_image(map, axis=True, show=True)

cens_all = find_centroids(map)
patterns_all, num_clusters_all = find_patterns(cens_all)

cluster_locs = {}
for pat in patterns_all:
    observed_points = pat[:, 0:2].tolist()
    cluster_index = identify_cluster(pat)
    cluster_locs[cluster_index] = observed_points

for fid in [20]:
    img = cv2.imread(f"jun_12_frame_{fid}.png")
    view_pose(img, cluster_locs, map, fid)
    show_image(img, axis=True, show=True)
