from jun_18_pattern_utils import *
import json

master_map = cv2.imread('jun17_map.png')
cens_all = find_centroids(master_map, visualize=True, area_threshold=10, dist_threshold=5)
patterns_all, num_clusters_all = find_patterns(cens_all)
visualize_patterns(master_map, patterns_all, num_clusters_all, fid=0)
cluster_locs = {}
for pat in patterns_all:
    observed_points = pat[:, 0:2].tolist()
    cluster_index = identify_cluster(pat)
    cluster_locs[cluster_index] = observed_points

# Save to file
with open('cluster_data.json', 'w') as f:
    json.dump(cluster_locs, f, indent=4)

# # Load back
# with open('data.json', 'r') as f:
#     loaded_data = json.load(f)