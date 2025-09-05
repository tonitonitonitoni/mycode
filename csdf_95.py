#Will implement a lot of improvements here.
#Goals:
# precompute the gradient field ✅
# separate optimizer penalty function calculation ✅
# change up viewpoint finding algorithm to sample from levelset directly ✅
# calculate norms with the gradient of the SDF, to compare to greedy direction finding algorithm ✅


import trimesh, casadi
import numpy as np, pinocchio as pin, matplotlib.pyplot as plt

from pinocchio import casadi as cpin

class CasadiSDF:
    #Class to create an SDF from a mesh for use in Casadi
    def __init__(self, mesh_path, grid_res=50, map_bound=5, debug=False):
        self.mesh = trimesh.load(mesh_path)
        self.grid_res = grid_res
        self.min_bound = np.array([-map_bound]*3) # bounds assume cubical volume centred at the origin 
        self.max_bound = np.array([map_bound]*3)
        self.voxel_size = (self.max_bound - self.min_bound) / (grid_res - 1)
        self.voxel_size_dm = casadi.DM(self.voxel_size)
        self.min_bound_dm = casadi.DM(self.min_bound)
        self.debug=debug
        self.boxes = None

    def generate(self):
        # Create 3D grid in WORLD COORDS
        grid_x = np.linspace(self.min_bound[0], self.max_bound[0], self.grid_res)
        grid_y = np.linspace(self.min_bound[1], self.max_bound[1], self.grid_res)
        grid_z = np.linspace(self.min_bound[2], self.max_bound[2], self.grid_res)

        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Signed distance at each world sample 
        signed_distances = -self.mesh.nearest.signed_distance(grid)  # shape: (N,)
        # Negative sign to make the distances positive outside and negative inside

        # Reshape to 3D tensor (x,y,z) with ij indexing
        sdf_np = np.reshape(signed_distances, (self.grid_res, self.grid_res, self.grid_res), order='C')

        gx = np.gradient(sdf_np, self.voxel_size[0], axis=0)
        gy = np.gradient(sdf_np, self.voxel_size[1], axis=1)
        gz = np.gradient(sdf_np, self.voxel_size[2], axis=2)
        grad_np = np.stack([gx, gy, gz], axis=-1)
        
        # Store axes for reference
        self.grid_x, self.grid_y, self.grid_z = grid_x, grid_y, grid_z

        # Build CasADi interpolants directly over WORLD axes.
        # CasADi expects values flattened in Fortran order for multi-D linear grids.
        values_F = sdf_np.flatten(order='F').tolist()
        self.sdf_interp = casadi.interpolant(
            'sdf', 'linear',
            [grid_x.tolist(), grid_y.tolist(), grid_z.tolist()],
            values_F
        )
        self.grad_interp = [
        casadi.interpolant(f'grad_{i}', 'linear',
                           [grid_x.tolist(), grid_y.tolist(), grid_z.tolist()],
                           grad_np[..., i].flatten(order='F').tolist())
        for i in range(3)
        ]
    
    def clamp_to_world(self, p):
        eps = 1e-9
        lo = self.min_bound_dm + eps
        hi = casadi.DM(self.max_bound) - eps
        return casadi.fmin(casadi.fmax(p, lo), hi)

    def eval_sdf(self, p):
        # Evaluate SDF at world coordinates, clamped to the grid bounds
        pw = self.clamp_to_world(p)
        return self.sdf_interp(casadi.vertcat(pw[0], pw[1], pw[2]))
    
    def grad_sdf(self, p): #New: precomputed once in generate method
        pw = self.clamp_to_world(p)
        g = [interp(casadi.vertcat(pw[0], pw[1], pw[2])) for interp in self.grad_interp]
        return casadi.vertcat(*g)
        
    def project_to_levelset(self, p, d0=0.5, tol = 0.01, alpha=0.1):
        """Project a point p onto the level set phi(x)=d0 using gradient descent."""
        p = np.array(p, dtype=float).reshape(3)
        phi = float(self.eval_sdf(p))
        while abs(phi - d0) > tol:
            g = np.array(self.grad_sdf(casadi.DM(p))).astype(float).reshape(-1)
            p = p - alpha * (phi - d0) * safe_normalize_num(g)
            phi = float(self.eval_sdf(p))
        return p

    def orbit_path_between(self, p_start, p_goal, d0=0.5, dt=0.02, v_tan=0.2,
                           k_n=0.1, orbit_axis=None, steps=5000, tol=5e-2,
                           k_g=0.5):
        """
        Generate an orbital path between two points on (or projected to) the level set.
        Adds a tangential goal-attraction term so the trajectory converges to `p_goal`:
            v(p) = v_tan * t(p) - k_n*(phi(p)-d0)*n(p) + k_g * Proj_T[(p_goal - p)]
        where Proj_T projects onto the tangent plane at p.

        Args:
            p_start, p_goal: (3,) numpy arrays.
            d0: desired level set value.
            dt: integration step.
            v_tan: nominal tangential speed magnitude.
            k_n: normal gain to regulate onto level set.
            orbit_axis: preferred axis to define a consistent tangent direction.
            steps: maximum number of RK4 steps.
            tol: stop when ||p - goal|| < tol.
            k_g: tangential goal attraction gain.
            cap_speed: if not None, clip ||v|| to this value (stability helper).
        Returns:
            (N,3) numpy array of waypoints.
        """
        # Project endpoints to the level set
        if self.debug: print(self.eval_sdf(p_start), self.eval_sdf(p_goal))
        if self.eval_sdf(p_start) - d0 > 0.1:
            p = self.project_to_levelset(p_start, d0=d0)
        else: p = np.array(p_start, dtype=float).reshape(3) 
        if self.eval_sdf(p_goal) - d0 > 0.1:
            goal = self.project_to_levelset(p_goal, d0=d0)
        else: goal = np.array(p_goal, dtype=float).reshape(3) 
        path = [p.copy()]
        path_length = 0

        def v_num(x):
            x = np.asarray(x, dtype=float).reshape(3)
            x_dm = casadi.DM(x)
            phi = float(self.sdf_interp(x_dm))
            # gradient and unit normal
            g = np.array(self.grad_sdf(x_dm)).astype(float).reshape(-1)
            n = safe_normalize_num(g)
            # tangent direction from orbit_axis
            a = np.array([0, 0, 1]) if orbit_axis is None else np.asarray(orbit_axis, dtype=float).reshape(3)
            t = np.cross(a, n)
            if np.linalg.norm(t) < 1e-12:
                t = np.cross(np.array([1, 0, 0]), n)
            t = safe_normalize_num(t)
            # tangential goal attraction: project (goal - x) onto tangent plane
            v_goal = goal - x
            v_goal_tan = v_goal - np.dot(v_goal, n) * n
            # compose velocity
            v = v_tan * t - k_n * (phi - d0) * n + k_g * v_goal_tan
            return v

        # RK4 integration
        iter=0
        while(np.linalg.norm(p - goal)) > tol:
            iter+=1
            if iter % 1000 ==0: tol *=5 
        #for k in range(steps):
            k1 = v_num(p)
            k2 = v_num(p + 0.5 * dt * k1)
            k3 = v_num(p + 0.5 * dt * k2)
            k4 = v_num(p + dt * k3)
            dp = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            p += dp
            path_length+=np.linalg.norm(dp)
            path.append(p.copy())
    
        return path, path_length

    def sample_surface(self, d0=0.5, N=1000, band=0.02, calc_norms=False):
        """
        Uniformly sample N points on the φ(x)=d0 isosurface.
        band: tolerance around d0 to accept a voxel as 'surface'
        """
        # Boolean mask of voxels near the surface
        sdf_vals = np.array([self.sdf_interp(casadi.DM([x,y,z])) 
                            for x in self.grid_x 
                            for y in self.grid_y 
                            for z in self.grid_z]).reshape(len(self.grid_x), len(self.grid_y), len(self.grid_z))

        mask = np.abs(sdf_vals - d0) < band
        surface_indices = np.argwhere(mask)

        samples = []
        if calc_norms: norms = []
        for _ in range(N):
            idx = surface_indices[np.random.randint(len(surface_indices))]
            i,j,k = idx
            # Randomly jitter inside the voxel cell
            jitter = np.random.rand(3) * self.voxel_size
            pt = np.array([self.grid_x[i], self.grid_y[j], self.grid_z[k]]) + jitter
            # Optional: refine to exactly φ(x)=d0
            pt = self.project_to_levelset(pt, d0=d0, tol=1e-3, alpha=0.05)
            samples.append(pt)
            if calc_norms: 
                n = np.array(self.grad_sdf(casadi.DM(pt))).astype(float).reshape(-1)
                n = safe_normalize_num(n)
                norms.append(-n)
        
        if calc_norms: return np.array(samples), np.array(norms)
        else: return np.array(samples)


    def add_boxes(self, boxes, pin_model):
        self.boxes = boxes
        self.pin_model = pin_model
        self._fk_boxes = {b.name: casadiBoxLocationFcn(self.pin_model, b) for b in self.boxes}
    
    def sdf_penalty(self, q, min_distance=0.25, k=20.0):
        # Get base position and orientation
        pos = q[0:3]           # base position
        quat = q[3:7]          # base orientation (xyzw)

        # Convert quaternion to rotation matrix
        Rm = quatxyzw_to_rot(quat)
       
        penalty = 0
        if self.boxes is not None:
            for b in self.boxes:
                offsets = box_offsets(b.half_extents)
                pos, Rm = self._fk_boxes[b.name](q[:self.pin_model.nq])
                for off in offsets:
                    corner = pos + Rm @ off
                    sdf_val = self.eval_sdf(corner)
                    penalty += smooth_sdf_penalty(sdf_val, min_distance, k=k)
        else:
            print("Bounding boxes not initialized, using base position only")
            sdf_val = self.eval_sdf(pos)
            penalty += smooth_sdf_penalty(sdf_val, min_distance, k=k)
        
        return penalty

    def sdf_violations(self, q, min_distance=0.25):
        """Return a vector of (sdf - min_distance) terms for multiple support points.
        These can be constrained to be >= 0 in the optimizer.
        """
        pos = q[0:3]
        quat = q[3:7]
        Rm = quatxyzw_to_rot(quat)

        terms = []
        if self.boxes is not None:
            for box in self.boxes:
                pos_world, Rm = self._fk_boxes[box.name](q[:self.pin_model.nq])

                offsets = self.box_support_offsets(box.half_extents)
                for off in offsets:
                    corner_world = pos_world + Rm @ (off)
                    sdf_val = self.evaluate_sdf(corner_world)
                    terms.append(sdf_val - min_distance)
        else:
            print("Bounding boxes not initialized, using base position only")
            sdf_val = self.eval_sdf(pos)
            terms.append(sdf_val - min_distance)

        return casadi.vertcat(*terms)

def smooth_sdf_penalty(sdf_val, d0=0.5, k=20.0):
    # penalize when sdf_val < min_clearance
    diff = d0 - sdf_val
    return (1.0 / k) * casadi.log(1 + casadi.exp(k * diff))

def quatxyzw_to_rot(q):
    x, y, z, w = q[0], q[1], q[2], q[3]

    row0 = [1 - 2 * (y**2 + z**2),     2 * (x*y - z*w),         2 * (x*z + y*w)]
    row1 = [2 * (x*y + z*w),           1 - 2 * (x**2 + z**2),   2 * (y*z - x*w)]
    row2 = [2 * (x*z - y*w),           2 * (y*z + x*w),         1 - 2 * (x**2 + y**2)]

    return casadi.vertcat(
        casadi.horzcat(*row0),
        casadi.horzcat(*row1),
        casadi.horzcat(*row2)
    )

def box_offsets(half_extents, midpoints=False):
        # Computes corners of bounding boxes
        hx, hy, hz = half_extents
        if midpoints: vals = [-1, 0, 1]
        else: vals = [-1, 1]
        offs = []
        for sx in vals:
            for sy in vals:
                for sz in vals:
                    if sx==0 and sy==0 and sz==0:
                        continue
                    offs.append([sx * hx, sy * hy, sz * hz])
        return [casadi.DM(o) for o in offs]

def casadiBoxLocationFcn(pin_model, box):
    # symbolic q
    cq = casadi.SX.sym("q", pin_model.nq, 1)

    # create symbolic data
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    # FK
    cpin.forwardKinematics(cmodel, cdata, cq)
    cpin.updateFramePlacements(cmodel, cdata)

    # body → world transform
    body_name = box.name.strip('_0')
    body_id = pin_model.getBodyId(body_name)
    M_world_body = cdata.oMf[body_id]  # symbolic SE3

    # constant SE3 for box offset in body frame (CasADi SX types)
    R_I = casadi.SX.eye(3)
    t_c = casadi.SX(box.center.reshape(3, 1)) if hasattr(box.center, 'reshape') else casadi.SX(box.center)
    if t_c.shape != (3, 1):
        t_c = casadi.reshape(t_c, 3, 1)
    box_in_body = cpin.SE3(R_I, t_c)

    # world transform of the box centre
    box_in_world = M_world_body * box_in_body

    centre_pos = box_in_world.translation
    rotmat = box_in_world.rotation

    return casadi.Function(f"box_{body_name}_loc", [cq], [centre_pos, rotmat])

def safe_normalize(v, eps=1e-8):
    return v / (casadi.norm_2(v) + eps)

def safe_normalize_num(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        print("zero")
        return v * 0.0
    return v / n

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def plot_orbit_path(path, p_start, p_goal, goal=None, mesh=None, show=True, ax=None):
    """Plot an orbit path in 3D, optionally overlaying the obstacle mesh."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        legend=True
    else: legend=False

    if goal is None:
        goal = path[-1,:]

    # Plot the orbit path
    ax.plot(path[:,0], path[:,1], path[:,2], 'r-', label="Orbit path")
    ax.scatter(path[0,0], path[0,1], path[0,2], c='g', marker='o', label="Start")
    ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='b', marker='x', label="End")
    ax.scatter(*goal, c='r', marker='*', label="Goal")

    #Plot the actual start and goal points
    ax.scatter(*p_start, c='g', marker='*', s=80, alpha=0.3, label="Actual Start")
    ax.scatter(*p_goal, c='b', marker='*', s=80, label="Actual Goal")

    if mesh is not None:
        # Plot mesh wireframe for context
        ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                        triangles=mesh.faces, color=(0.7,0.7,0.7,0.3), linewidth=0.2, edgecolor="gray")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if legend: ax.legend()
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    if show:
        plt.show()
    return ax
