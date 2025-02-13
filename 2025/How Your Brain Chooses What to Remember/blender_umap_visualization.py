import bpy
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_base_points(n_points=200, radius=3.0, noise_scale=0.1):
    """Generate initial 2D points with some noise."""
    # Generate random points in polar coordinates with more density towards the middle
    rho = np.sqrt(np.random.uniform(0, radius**2, n_points))  # Square root for uniform area density
    theta = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to cartesian coordinates
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    z = np.zeros_like(x)
    
    # Add small random noise to x and y
    noise = np.random.normal(0, noise_scale, (n_points, 3))
    noise[:, 2] = 0  # No noise in z direction for base points
    
    points_2d = np.column_stack([x, y, z]) + noise
    return points_2d

def apply_3d_warping(points_2d, height_scale=0.5, twist_scale=1.5):
    """Apply 3D warping transformation to 2D points."""
    points_3d = points_2d.copy()
    x, y = points_2d[:, 0], points_2d[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Apply warping in z direction
    z = height_scale * (x**2 - y**2) / (3.0 + r)  # Added r to denominator for smoother warping
    z += twist_scale * np.sin(3 * theta) * r/3.0  # Scale twist by radius for smoother effect
    
    points_3d[:, 2] = z

    # Add noise in all directions
    noise = np.random.normal(0, 0.1, (len(points_3d), 3))
    points_3d += noise

    return points_3d

def create_point_mesh(location, size=0.08):
    """Create a sphere representing a point."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    sphere = bpy.context.active_object
    
    # Add material
    material = bpy.data.materials.new(name="SphereMaterial")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.5, 0.8, 1)
    sphere.data.materials.append(material)
    
    return sphere

def create_geometry_nodes(spheres):
    """Create geometry nodes modifier with Object Info and Curve Line nodes."""
    # Create a new empty mesh object
    mesh = bpy.data.meshes.new("NodeLines")
    node_lines = bpy.data.objects.new("NodeLines", mesh)
    bpy.context.scene.collection.objects.link(node_lines)
    
    # Add geometry nodes modifier
    modifier = node_lines.modifiers.new(name="Edges", type='NODES')
    
    # Create a new node group
    node_group = bpy.data.node_groups.new("SphereEdges", "GeometryNodeTree")
    modifier.node_group = node_group
    
    # Set up the group input/output
    group_in = node_group.nodes.new('NodeGroupInput')
    group_out = node_group.nodes.new('NodeGroupOutput')
    group_in.location = (-200, 0)
    group_out.location = (1000, 0)
    
    # Add output socket
    node_group.interface.clear()
    node_group.interface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry'
    )
    
    # Create Object Info node for each sphere
    info_nodes = []
    for i, sphere in enumerate(spheres):
        info = node_group.nodes.new('GeometryNodeObjectInfo')
        info.location = (0, i * -50)
        info.transform_space = 'RELATIVE'
        info.inputs["Object"].default_value = sphere
        info_nodes.append(info)
    
    # Compute edges using nearest neighbors
    sphere_positions = np.array([sphere.location for sphere in spheres])
    nbrs = NearestNeighbors(n_neighbors=8).fit(sphere_positions)
    distances, indices = nbrs.kneighbors(sphere_positions)
    
    # Create Curve Line nodes for each edge
    curve_nodes = []
    x_offset = 200
    y_offset = 0
    
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip first neighbor (self)
            if i < j:  # Avoid duplicate edges
                curve_line = node_group.nodes.new('GeometryNodeCurvePrimitiveLine')
                curve_line.location = (x_offset, y_offset)
                
                node_group.links.new(info_nodes[i].outputs["Location"], curve_line.inputs["Start"])
                node_group.links.new(info_nodes[j].outputs["Location"], curve_line.inputs["End"])
                
                curve_nodes.append(curve_line)
                y_offset -= 50
    
    # Create Join Geometry node
    join_geometry = node_group.nodes.new('GeometryNodeJoinGeometry')
    join_geometry.location = (400, 0)
    
    # Connect all curve nodes to Join Geometry
    for curve_node in curve_nodes:
        node_group.links.new(curve_node.outputs["Curve"], join_geometry.inputs[0])
    
    # Create Curve Circle node
    curve_circle = node_group.nodes.new('GeometryNodeCurvePrimitiveCircle')
    curve_circle.location = (400, -200)
    curve_circle.inputs["Radius"].default_value = 0.02
    curve_circle.inputs["Resolution"].default_value = 32
    
    # Create Curve to Mesh node
    curve_to_mesh = node_group.nodes.new('GeometryNodeCurveToMesh')
    curve_to_mesh.location = (600, 0)
    
    # Connect nodes
    node_group.links.new(join_geometry.outputs[0], curve_to_mesh.inputs["Curve"])
    node_group.links.new(curve_circle.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    node_group.links.new(curve_to_mesh.outputs["Mesh"], group_out.inputs[0])
    
    return node_lines, node_group, info_nodes, curve_nodes



def create_animation(spheres, points_2d, points_3d, n_frames=120):
    """Create animation from 3D to 2D positions."""
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = n_frames
    
    def smooth_step(t):
        """Smooth step function for easing."""
        return t * t * (3 - 2 * t)
    
    # Set keyframes for each sphere
    for sphere, start_pos, end_pos in zip(spheres, points_3d, points_2d):
        for frame in range(n_frames + 1):
            bpy.context.scene.frame_set(frame)
            t = smooth_step(frame / n_frames)
            current_pos = start_pos * (1 - t) + end_pos * t
            sphere.location = current_pos
            sphere.keyframe_insert(data_path="location")

    # Add easing to keyframes
    for sphere in spheres:
        fcurves = sphere.animation_data.action.fcurves
        for fcurve in fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'BEZIER'
                kf.easing = 'AUTO'



def setup_scene():
    """Set up scene with lighting and camera."""
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Add camera
    bpy.ops.object.camera_add(location=(10, 10, 7))
    camera = bpy.context.active_object
    camera.rotation_euler = (0.9, 0, 2.356)
    bpy.context.scene.camera = camera
    
    # Add lights
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    bpy.context.active_object.data.energy = 3
    
    # Add ambient light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
    ambient = bpy.context.active_object
    ambient.data.energy = 100
    ambient.scale = (10, 10, 10)



def add_noise_fcurve_modifiers(objects):
    """Add noise modifiers to fcurves of all objects."""
    for i,obj in enumerate(objects):
        fcurves = obj.animation_data.action.fcurves
        for fcurve in fcurves:
            modifier = fcurve.modifiers.new(type='NOISE')
            modifier.strength = 0.2
            modifier.scale = 8
            
            # Change blend type to multiply
            modifier.blend_type = 'ADD'
            modifier.phase = i


            # Restrict frame range with smooth falloff
            # Enable frame range restriction
            modifier.use_restricted_range = True

            # Set frame range
            modifier.frame_start = 10
            modifier.frame_end = 110.0

            # Set blending
            modifier.blend_in = 20
            modifier.blend_out = 20



def main():
    # Set up scene
    setup_scene()
    
    # Generate points
    n_points = 300
    points_2d = generate_base_points(n_points=n_points)
    points_3d = apply_3d_warping(points_2d)
    
    # Create spheres at initial 2D positions
    spheres = []
    for point in points_2d:
        sphere = create_point_mesh(point)
        spheres.append(sphere)
    
    # Create geometry nodes setup for edges
    node_lines, node_group, info_nodes, curve_nodes = create_geometry_nodes(spheres)
    
    # Add material to edges
    edge_material = bpy.data.materials.new(name="EdgeMaterial")
    edge_material.use_nodes = True
    edge_material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
    node_lines.data.materials.append(edge_material)
    
    # Create animation
    create_animation(spheres, points_2d, points_3d)
    print("Animation setup complete")
    
    add_noise_fcurve_modifiers(spheres)
    print("Noise modifiers added")
    



main()