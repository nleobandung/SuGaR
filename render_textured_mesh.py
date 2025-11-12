#!/usr/bin/env python3
"""
Render a textured SuGaR mesh and save as PNG.

This script loads a textured mesh (OBJ file) and renders it from a specific camera view,
saving the result as a PNG image.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
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
from sugar_utils.general_utils import str2bool


def main():
    parser = argparse.ArgumentParser(
        description='Render a textured SuGaR mesh and save as PNG.'
    )
    
    # Required arguments
    parser.add_argument(
        '-m', '--mesh_path',
        type=str,
        required=True,
        help='Path to the textured mesh OBJ file.'
    )
    parser.add_argument(
        '--cam_idx',
        type=int,
        default=125,
        help='Camera index to render from.'
    )
    
    # Scene and checkpoint (needed to load cameras)
    parser.add_argument(
        '-s', '--scene_path',
        type=str,
        required=True,
        help='Path to the scene data (directory containing images subdirectory).'
    )
    parser.add_argument(
        '-c', '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the vanilla 3D Gaussian Splatting checkpoint directory.'
    )
    
    # Optional arguments
    parser.add_argument(
        '-i', '--iteration_to_load',
        type=int,
        default=7000,
        help='Iteration to load from checkpoint. Default: 7000'
    )
    parser.add_argument(
        '-o', '--output_path',
        type=str,
        default=None,
        help='Output path for the rendered PNG. If not provided, will be saved next to mesh file.'
    )
    parser.add_argument(
        '--white_background',
        type=str2bool,
        default=True,
        help='Use white background instead of black. Default: True'
    )
    parser.add_argument(
        '--eval',
        type=str2bool,
        default=False,
        help='Use eval split for cameras. Default: False'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device index. Default: 0'
    )
    parser.add_argument(
        '--image_height',
        type=int,
        default=None,
        help='Image height for rendering. If not provided, will use camera height.'
    )
    parser.add_argument(
        '--image_width',
        type=int,
        default=None,
        help='Image width for rendering. If not provided, will use camera width.'
    )
    
    args = parser.parse_args()
    
    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    print("=" * 60)
    print("Loading cameras from scene and checkpoint...")
    print("=" * 60)
    
    # Load cameras from the scene
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.scene_path,
        output_path=args.checkpoint_path,
        iteration_to_load=args.iteration_to_load,
        load_gt_images=False,
        eval_split=args.eval,
        eval_split_interval=8,
    )
    
    cameras_to_use = nerfmodel.training_cameras
    if args.eval:
        cameras_to_use = nerfmodel.test_cameras
    
    print(f'Loaded {len(cameras_to_use.gs_cameras)} cameras.')
    
    # Validate camera index
    if args.cam_idx < 0 or args.cam_idx >= len(cameras_to_use.gs_cameras):
        raise ValueError(
            f'Camera index {args.cam_idx} is out of range. '
            f'Valid range: 0-{len(cameras_to_use.gs_cameras) - 1}'
        )
    
    print(f"Using camera index {args.cam_idx}")
    print(f"Image name: {cameras_to_use.gs_cameras[args.cam_idx].image_name}")
    
    print("=" * 60)
    print(f"Loading textured mesh from {args.mesh_path}...")
    print("=" * 60)
    
    # Load the textured mesh
    if not os.path.exists(args.mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_path}")
    
    textured_mesh = load_objs_as_meshes([args.mesh_path]).to(device)
    print(
        f"Loaded textured mesh with {len(textured_mesh.verts_list()[0])} vertices "
        f"and {len(textured_mesh.faces_list()[0])} faces."
    )
    
    # Determine image dimensions
    if args.image_height is None or args.image_width is None:
        # Get dimensions from camera
        cam = cameras_to_use.p3d_cameras[args.cam_idx]
        image_height = int(cam.height[0].item()) if hasattr(cam, 'height') else 800
        image_width = int(cam.width[0].item()) if hasattr(cam, 'width') else 800
    else:
        image_height = args.image_height
        image_width = args.image_width
    
    print(f"Rendering at resolution: {image_width} x {image_height}")
    
    # Set up rasterization and rendering
    faces_per_pixel = 1
    max_faces_per_bin = 50_000
    
    mesh_raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )
    
    # Background color
    bg_color = (1.0, 1.0, 1.0) if args.white_background else (0.0, 0.0, 0.0)
    
    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(
        cameras=cameras_to_use.p3d_cameras[args.cam_idx],
        raster_settings=mesh_raster_settings,
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device,
            cameras=cameras_to_use.p3d_cameras[args.cam_idx],
            lights=lights,
            blend_params=BlendParams(background_color=bg_color),
        )
    )
    
    print("=" * 60)
    print("Rendering mesh...")
    print("=" * 60)
    
    # Render the mesh
    with torch.no_grad():
        p3d_cameras = cameras_to_use.p3d_cameras[args.cam_idx]
        rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
        rgb_img = rgb_img.clamp(min=0, max=1)
    
    # Convert to numpy and save
    rgb_np = rgb_img.cpu().numpy()
    
    # Determine output path
    if args.output_path is None:
        mesh_dir = os.path.dirname(args.mesh_path)
        mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
        output_path = os.path.join(mesh_dir, f"{mesh_name}_cam{args.cam_idx}.png")
    else:
        output_path = args.output_path
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as PNG
    rgb_uint8 = (rgb_np * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    img.save(output_path)
    
    print("=" * 60)
    print(f"Rendered image saved to: {output_path}")
    print("=" * 60)
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

