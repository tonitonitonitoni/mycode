import tqdm
from csdf_95 import *
import dill as pickle

def sphericalVectorArray(nray1=20, nray2=20):
    vec_list = []

    num_phi_steps = nray1  # Azimuthal steps (around Z-axis)
    num_theta_steps = nray2 # Inclination steps (from +Z to -Z)

    # Total rays = num_phi_steps * num_theta_steps = 36 * 20 = 720
    nray = nray1 * nray2
    # Generate phi (azimuth) angles from 0 to 2*pi (exclusive of 2*pi)
    phi_angles = np.linspace(0, 2 * np.pi, num_phi_steps, endpoint=False)
    # Generate theta (inclination) angles from 0 to pi (inclusive of poles)
    theta_angles = np.linspace(0, np.pi, num_theta_steps, endpoint=True)

    for phi in phi_angles:
        for theta in theta_angles:
            # Convert spherical coordinates (r=1, theta, phi) to Cartesian (x, y, z)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            vec_list.append([x, y, z])

    vec = np.array(vec_list, dtype=np.float64)

    return vec

#lookat vector to rotation matrix
def matrixFromVector(direction):
    d = np.array(direction)
    if np.linalg.norm(d) == 0:
        r = np.eye(3)
    else:
        d = d / np.linalg.norm(d)
        z = np.array([-d[1], d[0], 0])
        n = np.cross(d, z)
        if abs(np.linalg.norm(n))<1e-8:
            z=np.array([-d[2], 0, d[0]])
            n = np.cross(d,z)
            if abs(np.linalg.norm(n))<1e-8:
                z = np.array([0, -d[2], d[1]])
                n = np.cross(d,z)
                if abs(np.linalg.norm(n))<1e-8:
                    print("eek! something is very wrong!")
        r = np.vstack([z, n, d]).T
    return r


#Next-Best-View trajectory finding functions
def valid_triangles(mesh, lookat, eye_pos, min_dist=0.1, max_dist=5.0, max_angle_deg=30):
    #create the mesh scene
    scene = mesh.scene()
    scene.camera.resolution = [80, 60]
    # Set up camera transform
    t_r = np.eye(4)
    t_r[0:3, -1] = eye_pos
    t_r[:3, :3] = matrixFromVector((lookat - eye_pos))
    scene.camera_transform = t_r

    # Use reduced resolution already set in scene.camera.resolution = [80, 60]
    origins, directions, _ = scene.camera_rays()

   # Use fast ray intersection: get first hit triangle and distances
    hit_points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    if len(index_tri) == 0:
        return {}
    ray_origins_hit = origins[index_ray]
    ray_dirs_hit = directions[index_ray]
    valid_tris = index_tri
    valid_dirs = ray_dirs_hit
    valid_dists = np.linalg.norm(hit_points - ray_origins_hit, axis=1)

    # Only keep rays that hit something
    valid_mask = index_tri != -1
    if not np.any(valid_mask):
        return {}

    # Filter by distance
    distance_mask = (valid_dists > min_dist) & (valid_dists < max_dist)
    if not np.any(distance_mask):
        return {}
    valid_tris = valid_tris[distance_mask]
    valid_dirs = valid_dirs[distance_mask]
    valid_dists = valid_dists[distance_mask]

    # Get triangle normals for all hit triangles
    triangle_normals = mesh.face_normals
    # Get normal for each hit
    hit_normals = triangle_normals[valid_tris]
    # Normalize directions
    dirs_norm = valid_dirs / (np.linalg.norm(valid_dirs, axis=1, keepdims=True) + 1e-12)
    # Compute cos(angle) between ray dir and triangle normal
    cos_angles = np.einsum('ij,ij->i', dirs_norm, hit_normals)
    angles_deg = np.degrees(np.arccos(np.clip(np.abs(cos_angles), 0, 1.0)))

    # Filter by angle
    angle_mask = angles_deg < max_angle_deg
    if not np.any(angle_mask):
        return {}
    valid_tris = valid_tris[angle_mask]
    
    return set(valid_tris)

def best_direction(mesh, viewpoint):
    best_score = -1
    best_direction = None
    #Get the vector of ray directions
    best_valid_tris = None
    dirs = sphericalVectorArray() 
    for dir in dirs:
        lookat = viewpoint + dir
        eye_pos = viewpoint
        valid_tris = valid_triangles(mesh, lookat, eye_pos)
        score = len(valid_tris)
        if score == 0:
            continue

        if score > best_score:
            best_score = score
            best_direction = dir
            best_valid_tris = valid_tris
            print(f"best score: {best_score}", end="\r")
        
    if best_direction is None: print("FML")
    return best_direction, best_valid_tris

pwd = '/Users/antoniahoffman/PycharmProjects/manipulatorProject'

#Load the robot model
mjcf_path = pwd +'/ur3/ur3_box_smaller.xml'
#Create Pinocchio model
pin_model, collision_model, visual_model = pin.buildModelsFromMJCF(mjcf_path)
pin_model.gravity = pin.Motion.Zero()
pin_data = pin_model.createData()

#load precomputed boxes 
name="ur3_box_smaller"
with open(f"aabb_list_{name}.pkl", "rb") as file:
    boxes = pickle.load(file)
    
#load the target satellite
model = 'simple'
#model = 'GRO'
mesh_path = pwd + f'/mujoco_ws/models/{model}_moved_scaled.stl'

#generate the SDF tensor
csdf = CasadiSDF(mesh_path, grid_res=30)
csdf.generate() 

#add the boxes
csdf.add_boxes(boxes, pin_model)
#
#pwd = '/Users/antoniahoffman/PycharmProjects/manipulatorProject'
#model = 'simple'
#n_vps = 32
#model = 'GRO'
#mesh_path = pwd + f'/mujoco_ws/models/{model}_moved_scaled.stl'

#mesh = trimesh.load(mesh_path)
#samples, vps, norms = trimesh_poisson_viewpoints(mesh, num_viewpoints = n_vps)
# with open('csdf_box_test.pkl', 'rb') as f:
#     csdf = pickle.load(f)

n_vps = 100
vps = csdf.sample_surface(N = n_vps)

greedy_directions = {}
greedy_triangles = {}
for i, vp in tqdm.tqdm(enumerate(vps), total=n_vps):
    greedy_directions[i], greedy_triangles[i]  = best_direction(csdf.mesh, vp)
    print(f"{len(greedy_triangles[i])} triangles found in best direction for vp {i}")
 
with open(f'directions_first_csdf.pkl', 'wb') as f:
    pickle.dump({'viewpoints': vps, 
                 'best directions': greedy_directions, 
                 'triangles': greedy_triangles}, f)