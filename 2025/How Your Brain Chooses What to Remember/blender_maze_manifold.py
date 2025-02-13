import pickle
import bpy
import numpy as np
import BlenderRat.BlenderRat.utils as utils
import matplotlib.pyplot as plt
import cmasher

def normalize_values(values_array):
    """Normalize array values to range [0,1]."""
    values_min, values_max = values_array.min(), values_array.max()
    return (values_array - values_min) / (values_max - values_min)

def create_base_mesh(points_array, values_normalized):
    """Create the base mesh with vertices and custom attributes."""
    mesh = bpy.data.meshes.new('ScatterPointsMesh')
    obj = bpy.data.objects.new('ScatterPoints', mesh)
    
    # Link object to scene
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Create vertices
    verts = [tuple(point) for point in points_array]
    mesh.from_pydata(verts, [], [])  # No edges or faces
    
    # Create custom attribute for colors
    value_attr = mesh.attributes.new(name="value_normalized", type='FLOAT', domain='POINT')
    value_attr.data.foreach_set('value', values_normalized)
    
    return obj

def srgb_to_linear(srgb):
    """Convert sRGB color values to linear color space."""
    # Handle numpy arrays or lists
    if hasattr(srgb, '__iter__'):
        return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in srgb]
    # Handle single values
    return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4


def create_material(cmap):
    """
    Create and set up the material for points visualization using the provided colormap.
    
    Args:
        cmap: matplotlib colormap object to use for coloring
    """
    material = bpy.data.materials.new(name="PointsMaterial")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    
    # Add material nodes
    mat_nodes = material.node_tree.nodes
    mat_output = mat_nodes.new('ShaderNodeOutputMaterial')
    mat_emission = mat_nodes.new('ShaderNodeEmission')
    mat_attribute = mat_nodes.new('ShaderNodeAttribute')
    color_ramp = mat_nodes.new('ShaderNodeValToRGB')
    
    # Configure nodes
    mat_attribute.attribute_name = "value_normalized"
    
    # Set color ramp interpolation to LINEAR for smooth gradient
    color_ramp.color_ramp.interpolation = 'LINEAR'
    
    # Clear default color stops and create new ones from colormap
    # Remove all elements except the first one
    while len(color_ramp.color_ramp.elements) > 1:
        color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[-1])
    
    # Sample colors from the colormap
    n_samples = 32
    
    # Set the first color (we already have this element)
    first_color = srgb_to_linear(cmap(0)[:3])
    print(first_color)

    color_ramp.color_ramp.elements[0].position = 0
    color_ramp.color_ramp.elements[0].color = first_color + [1.0]
    
    # Add the rest of the colors
    for i in range(1, n_samples):
        pos = i / (n_samples - 1)
        color = srgb_to_linear(cmap(pos)[:3]) 
        element = color_ramp.color_ramp.elements.new(pos)
        element.color = color + [1.0]
    
    # Position nodes
    mat_output.location = (400, 0)
    mat_emission.location = (200, 0)
    color_ramp.location = (0, 0)
    mat_attribute.location = (-200, 0)
    
    # Link material nodes
    links = material.node_tree.links
    links.new(mat_emission.outputs['Emission'], mat_output.inputs['Surface'])
    links.new(color_ramp.outputs['Color'], mat_emission.inputs['Color'])
    links.new(mat_attribute.outputs['Fac'], color_ramp.inputs['Fac'])
    
    return material


def setup_geometry_nodes(obj, material, sphere_radius=0.05):
    """Set up the geometry nodes system for point visualization."""
    # Add Geometry Nodes modifier
    geo_nodes = obj.modifiers.new(name="Scatter", type='NODES')
    node_group = bpy.data.node_groups.new('ScatterGeometry', 'GeometryNodeTree')
    geo_nodes.node_group = node_group
    
    # Create nodes
    nodes = node_group.nodes
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')
    mesh_to_points = nodes.new('GeometryNodeMeshToPoints')
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    realize_instances = nodes.new('GeometryNodeRealizeInstances')
    uv_sphere = nodes.new('GeometryNodeMeshUVSphere')
    set_material = nodes.new('GeometryNodeSetMaterial')
    
    # Configure sphere properties
    uv_sphere.inputs['Radius'].default_value = sphere_radius
    uv_sphere.inputs['Segments'].default_value = 10
    uv_sphere.inputs['Rings'].default_value = 10
    
    # Position nodes
    group_input.location = (-400, 0)
    mesh_to_points.location = (-200, 0)
    instance_on_points.location = (0, 0)
    set_material.location = (200, 0)
    group_output.location = (400, 0)
    uv_sphere.location = (-200, -200)
    
    # Set up node group interface
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # Connect nodes
    links = node_group.links
    links.new(group_input.outputs[0], mesh_to_points.inputs['Mesh'])
    links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(uv_sphere.outputs['Mesh'], instance_on_points.inputs['Instance'])
    links.new(instance_on_points.outputs['Instances'], realize_instances.inputs['Geometry'])
    links.new(realize_instances.outputs['Geometry'], set_material.inputs['Geometry'])
    links.new(set_material.outputs['Geometry'], group_output.inputs[0])
    
    # Set material
    set_material.inputs['Material'].default_value = material

def create_scatter_plot(points_array, values_array, cmap, sphere_radius=0.05):
    """
    Create a 3D scatter plot in Blender using points and geometry nodes.
    
    Args:
        points_array: numpy array of shape (N,3) with point coordinates
        values_array: numpy array of shape (N,) with values to map to colors
        cmap: matplotlib colormap object
        sphere_radius: radius of the spheres
    
    Returns:
        bpy.types.Object: The created scatter plot object
    """
    # Normalize values
    values_normalized = normalize_values(values_array)
    
    # Create base mesh with vertices
    obj = create_base_mesh(points_array, values_normalized)
    
    # Create and setup material
    material = create_material(cmap)
    
    # Setup geometry nodes system
    setup_geometry_nodes(obj, material, sphere_radius)
    
    return obj

def main():
    data = pickle.load(open('/Users/artemkirsanov/YouTube/SWR memory selection/Code/unsupervised_UMAP_data.pickle', 'rb')) # Change path to the pickle file 

    manifold = data['embedding'][:, :3]  # (n_points, 3) array
    
    # Create scatter plot
    cmap = cmasher.get_sub_cmap(cmasher.cosmic, 0.2, 1)
    points_obj = create_scatter_plot(manifold, data['lin_pos_sm'], cmap=cmap, sphere_radius=0.05)
    print('Scatter plot created')
    
    # Create curve
    curve_xyz = manifold[data['target'].reshape(-1) == 10, :]
    utils.build_curve_from_array(curve_xyz) # Uncomment to create curve
    print('Curve created')


main()