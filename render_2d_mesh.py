import argparse
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )
from pytorch3d.renderer.blending import BlendParams
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import load_refined_model
from sugar_utils.general_utils import str2bool

def render_mesh(mesh_path, cam_idx, save_dir=None, scene_name=None):
    """Render a mesh from a specific camera view."""
    
    # Extract scene name from mesh path if not provided
    if scene_name is None:
        # Find the path component after 'refined_mesh/' or 'coarse_mesh/'
        path_parts = mesh_path.split('/')
        try:
            mesh_idx = path_parts.index('refined_mesh') if 'refined_mesh' in path_parts else path_parts.index('coarse_mesh')
            scene_name = path_parts[mesh_idx + 1]
        except (ValueError, IndexError):
            scene_name = path_parts[-3]  # Fallback to third-to-last component
    
    # Determine source path and GS checkpoint path
    scene_mapping = {
        'garden': {
            'source_path': '../data/garden/',
            'gs_checkpoint_path': './output/vanilla_gs/garden/'
        },
        'counter': {
            'source_path': '../data/counter/',
            'gs_checkpoint_path': './output/vanilla_gs/counter/'
        }
    }
    
    # Find matching scene
    source_path = None
    gs_checkpoint_path = None
    
    for key, paths in scene_mapping.items():
        if key in scene_name:
            source_path = paths['source_path']
            gs_checkpoint_path = paths['gs_checkpoint_path']
            break
    
    if source_path is None:
        # Try common patterns
        for key in ['garden', 'counter', 'kitchen', 'room', 'bicycle', 'bonsai', 'stump', 'train', 'truck']:
            if key in scene_name:
                source_path = f'../data/{key}/'
                gs_checkpoint_path = f'./output/vanilla_gs/{key}/'
                print(f'Attempting to use scene: {key}')
                break
    
    if source_path is None or gs_checkpoint_path is None:
        raise ValueError(f"Unknown scene: {scene_name}. Please add it to the scene_mapping in render_2d_mesh.py")
    
    # ==================== Load NeRF model and training data ====================
    print(f"\nLoading config {gs_checkpoint_path}...")
    iteration_to_load = 7000
    
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,
        eval_split=False,
        eval_split_interval=8,
    )
    
    print(f'{len(nerfmodel.training_cameras)} training images detected.')
    print(f'The model has been trained for {iteration_to_load} steps.')
    print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")
    
    cameras_to_use = nerfmodel.training_cameras
    
    # ==================== Load mesh ====================
    print(f"\nLoading refined mesh from {mesh_path}, this could take a minute...")
    textured_mesh = load_objs_as_meshes([mesh_path]).to(nerfmodel.device)
    print(f"Loaded textured mesh with {len(textured_mesh.verts_list()[0])} vertices and {len(textured_mesh.faces_list()[0])} faces.")
    
    # ==================== Load refined_sugar for device/height info ====================
    # Derive refined sugar path from mesh path
    mesh_dir = os.path.dirname(mesh_path)
    mesh_filename = os.path.basename(mesh_path)
    
    # Try to find the corresponding .pt file
    refined_sugar_path = None
    possible_parent_dirs = [
        os.path.join(mesh_dir, "2000.pt"),
        os.path.join(os.path.dirname(mesh_dir), "sugarfine_" + mesh_filename.replace('.obj', '')[-30:].replace('gaussperface1', '') + '/2000.pt'),
    ]
    
    # Search for .pt files in parent directory structure
    pt_files = glob.glob(os.path.join(mesh_dir.replace('refined_mesh', 'refined'), '**', '*.pt'), recursive=True)
    if pt_files:
        refined_sugar_path = pt_files[0]
    
    print(f"Searching for refined model near: {mesh_path}")
    print(f"Found refined model: {refined_sugar_path}")
    
    # Use refined_sugar if available, otherwise fall back to nerfmodel
    if refined_sugar_path and os.path.exists(refined_sugar_path):
        refined_sugar = load_refined_model(refined_sugar_path, nerfmodel)
        device_to_use = refined_sugar.device
        image_height = refined_sugar.image_height
        image_width = refined_sugar.image_width
    else:
        device_to_use = nerfmodel.device
        image_height = nerfmodel.image_height
        image_width = nerfmodel.image_width
    
    # ==================== Set up renderer ====================
    faces_per_pixel = 1
    
    mesh_raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
    )
    
    lights = AmbientLights(device=device_to_use)
    rasterizer = MeshRasterizer(
        cameras=cameras_to_use.p3d_cameras[cam_idx], 
        raster_settings=mesh_raster_settings,
    )
    
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device_to_use, 
            cameras=cameras_to_use.p3d_cameras[cam_idx],
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
    )
    
    # ==================== Render ====================
    with torch.no_grad():    
        print(f"Rendering image {cam_idx}")
        print("Image ID:", cameras_to_use.gs_cameras[cam_idx].image_name)
        
        p3d_cameras = cameras_to_use.p3d_cameras[cam_idx]
        rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
    
    # ==================== Save or display ====================
    plot_ratio = 2.
    plt.figure(figsize=(10 * plot_ratio, 10 * plot_ratio))
    plt.axis("off")
    plt.title(f"Rendered mesh - View {cam_idx} - {cameras_to_use.gs_cameras[cam_idx].image_name}")
    plt.imshow(rgb_img.cpu().numpy())
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        mesh_filename = os.path.basename(mesh_path).replace('.obj', '')
        output_path = os.path.join(save_dir, f"{mesh_filename}_view{cam_idx}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=100, pad_inches=0)
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a textured mesh from different camera views.')
    
    parser.add_argument('--mesh_path', type=str, required=True,
                        help='Path to the .obj mesh file to render.')
    
    parser.add_argument('--cam_idx', type=int, default=8,
                        help='Camera index to render from. Default is 8.')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory to save rendered images. If None, will be set to output/refined_mesh/<scene_name>/2D_visualizations/')
    
    parser.add_argument('--display', type=str2bool, default=False,
                        help='Display the rendered image using matplotlib. Default is False.')
    
    args = parser.parse_args()
    
    # Extract scene name from mesh path for output directory
    path_parts = args.mesh_path.split('/')
    try:
        mesh_idx = path_parts.index('refined_mesh') if 'refined_mesh' in path_parts else path_parts.index('coarse_mesh')
        scene_name = path_parts[mesh_idx + 1]
    except (ValueError, IndexError):
        scene_name = path_parts[-3]  # Fallback to third-to-last component
    
    # Determine save directory
    if args.output_dir is None:
        args.output_dir = os.path.join('output/refined_mesh', scene_name, '2D_visualizations')
    
    # Render the mesh
    render_mesh(
        mesh_path=args.mesh_path,
        cam_idx=args.cam_idx,
        save_dir=args.output_dir,
        scene_name=scene_name
    )
    
    # Display if requested
    if args.display:
        plt.show()
