import os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss

from rich.console import Console
import time


def coarse_training_without_reg_and_entropy(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    use_eval_split = True
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    initialize_from_trained_3dgs = True  # True or False
    if initialize_from_trained_3dgs:
        prune_at_start = False
        start_pruning_threshold = 0.5
    no_rendering = freeze_gaussians

    n_points_at_start = None  # If None, takes all points in the SfM point cloud

    learnable_positions = True  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 4  

        
    # -----Radiance Mesh-----
    triangle_scale=1.
    
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = True  # TODO: Try True

        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = 15_000  # Changed

    spatial_lr_scale = None
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
        
    # Densifier and pruning
    heavy_densification = False
    if initialize_from_trained_3dgs:
        densify_from_iter = 500 + 99999 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 - 7000 # 7000
    else:
        densify_from_iter = 500 # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 # 7000

    if heavy_densification:
        densification_interval = 50  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
    else:
        densification_interval = 100  # 100
        opacity_reset_interval = 3000  # 3000
        
        densify_grad_threshold = 0.0002  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01

    # Data processing and batching
    n_images_to_use_for_training = -1  # If -1, uses all images

    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # No regularization (entropy and SDF regularization disabled)
    regularize = False
    regularity_knn = 0
    beta_mode = None
        
    # Opacity management
    prune_low_opacity_gaussians_at = []
    prune_hard_opacity_threshold = 0.5

    # Warmup
    do_resolution_warmup = False
    if do_resolution_warmup:
        resolution_warmup_every = 500
        current_resolution_factor = downscale_resolution_factor * 4.
    else:
        current_resolution_factor = downscale_resolution_factor

    do_sh_warmup = True  # Should be True
    if initialize_from_trained_3dgs:
        do_sh_warmup = False
        sh_levels = 4  # nerfmodel.gaussians.active_sh_degree + 1
        CONSOLE.print("Changing sh_levels to match the loaded model:", sh_levels)
    if do_sh_warmup:
        sh_warmup_every = 1000
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels
        

    # -----Log and save-----
    print_loss_every_n_iterations = 200
    save_model_every_n_iterations = 1_000_000
    # save_milestones = [9000, 12_000, 15_000]
    save_milestones = [15_000]

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-2])
            
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load    
    
    sugar_checkpoint_path = f'sugarcoarse_3Dgs{iteration_to_load}_without_reg_and_entropy/'
    sugar_checkpoint_path = os.path.join(args.output_dir, sugar_checkpoint_path)
    
    use_eval_split = args.eval
    use_white_background = args.white_background
    
    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("SUGAR checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Output directory:", args.output_dir)
    CONSOLE.print("Regularization: None (without regularization)")
    CONSOLE.print("Training mode: Coarse without regularization")
    CONSOLE.print("Eval split:", use_eval_split)
    CONSOLE.print("White background:", use_white_background)
    CONSOLE.print("---------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        white_background=use_white_background,
        )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )

    # Point cloud
    if initialize_from_trained_3dgs:
        with torch.no_grad():    
            print("Initializing model from trained 3DGS...")
            with torch.no_grad():
                sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))
            
            from sugar_utils.spherical_harmonics import SH2RGB
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
            if prune_at_start:
                with torch.no_grad():
                    start_prune_mask = nerfmodel.gaussians.get_opacity.view(-1) > start_pruning_threshold
                    points = points[start_prune_mask]
                    colors = colors[start_prune_mask]
            n_points = len(points)
    else:
        CONSOLE.print("\nLoading SfM point cloud...")
        pcd = fetchPly(ply_path)
        points = torch.tensor(pcd.points, device=nerfmodel.device).float().cuda()
        colors = torch.tensor(pcd.colors, device=nerfmodel.device).float().cuda()
    
        if n_points_at_start is not None:
            n_points = n_points_at_start
            pts_idx = torch.randperm(len(points))[:n_points]
            points, colors = points.to(device)[pts_idx], colors.to(device)[pts_idx]
        else:
            n_points = len(points)
            
    CONSOLE.print(f"Point cloud generated. Number of points: {len(points)}")
        
    # Background tensor if needed
    if use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = None
    
    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=points,
        colors=colors,
        initialize=True,
        sh_levels=sh_levels,
        learnable_positions=learnable_positions,
        triangle_scale=triangle_scale,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode=None,
        freeze_gaussians=freeze_gaussians,
        surface_mesh_to_bind=None,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=False,
        learn_surface_mesh_opacity=False,
        learn_surface_mesh_scales=False,
        n_gaussians_per_surface_triangle=1,
        )
    
    if initialize_from_trained_3dgs:
        with torch.no_grad():            
            CONSOLE.print("Initializing 3D gaussians from 3D gaussians...")
            if prune_at_start:
                sugar._scales[...] = nerfmodel.gaussians._scaling.detach()[start_prune_mask]
                sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()[start_prune_mask]
                sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()[start_prune_mask]
                sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()[start_prune_mask]
                sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()[start_prune_mask]
            else:
                sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
                sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
                sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
                sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
                sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
        
    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
 
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()
    
    
    # ====================Initialize optimizer====================
    if spatial_lr_scale is None:
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])
        
        
    # ====================Initialize densifier====================
    gaussian_densifier = SuGaRDensifier(
        sugar_model=sugar,
        sugar_optimizer=optimizer,
        max_grad=densify_grad_threshold,
        min_opacity=prune_opacity_threshold,
        max_screen_size=densify_screen_size_threshold,
        scene_extent=cameras_spatial_extent,
        percent_dense=densification_percent_distinction,
        )
    CONSOLE.print("Densifier initialized.")
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')
    
    
    # ====================Start training====================
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
    
    if initialize_from_trained_3dgs:
        iteration = 7000 - 1
    
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)
        
        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1
            
            # Update learning rates
            optimizer.update_learning_rate(iteration)
            
            # Prune low-opacity gaussians
            if (iteration-1) in prune_low_opacity_gaussians_at:
                CONSOLE.print("\nPruning gaussians with low-opacity for further optimization...")
                prune_mask = (gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
                gaussian_densifier.prune_points(prune_mask)
                CONSOLE.print(f'Pruning finished: {sugar.n_points} gaussians left.')
            
            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            
            # Computing rgb predictions
            if not no_rendering:
                outputs = sugar.render_image_gaussian_rasterizer( 
                    camera_indices=camera_indices.item(),
                    verbose=False,
                    bg_color = bg_tensor,
                    sh_deg=current_sh_levels-1,
                    sh_rotations=None,
                    compute_color_in_rasterizer=compute_color_in_rasterizer,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                    return_opacities=False,
                    )
                pred_rgb = outputs['image'].view(-1, 
                    sugar.image_height, 
                    sugar.image_width, 
                    3)
                radii = outputs['radii']
                viewspace_points = outputs['viewspace_points']
                
                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
                
                # Gather rgb ground truth
                gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)           
                gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                    
                # Compute loss 
                loss = loss_fn(pred_rgb, gt_rgb)
            else:
                # No rendering (freeze_gaussians mode)
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Update parameters
            loss.backward()
            
            # Densification
            with torch.no_grad():
                if (not no_rendering) and (iteration < densify_until_iter):
                    gaussian_densifier.update_densification_stats(viewspace_points, radii, visibility_filter=radii>0)

                    if iteration > densify_from_iter and iteration % densification_interval == 0:
                        size_threshold = gaussian_densifier.max_screen_size if iteration > opacity_reset_interval else None
                        gaussian_densifier.densify_and_prune(densify_grad_threshold, prune_opacity_threshold, 
                                                    cameras_spatial_extent, size_threshold)
                        CONSOLE.print("Gaussians densified and pruned. New number of gaussians:", len(sugar.points))
                    
                    if iteration % opacity_reset_interval == 0:
                        gaussian_densifier.reset_opacity()
                        CONSOLE.print("Opacity reset.")
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if iteration==1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                    "computed in", (time.time() - t0) / 60., "minutes.")
                with torch.no_grad():
                    scales = sugar.scaling.detach()
             
                    CONSOLE.print("------Stats-----")
                    CONSOLE.print("---Min, Max, Mean, Std")
                    CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(), sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                    CONSOLE.print("Scaling factors:", sugar.scaling.min().item(), sugar.scaling.max().item(), sugar.scaling.mean().item(), sugar.scaling.std().item(), sep='   ')
                    CONSOLE.print("Quaternions:", sugar.quaternions.min().item(), sugar.quaternions.max().item(), sugar.quaternions.mean().item(), sugar.quaternions.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates dc:", sugar._sh_coordinates_dc.min().item(), sugar._sh_coordinates_dc.max().item(), sugar._sh_coordinates_dc.mean().item(), sugar._sh_coordinates_dc.std().item(), sep='   ')
                    CONSOLE.print("Sh coordinates rest:", sugar._sh_coordinates_rest.min().item(), sugar._sh_coordinates_rest.max().item(), sugar._sh_coordinates_rest.mean().item(), sugar._sh_coordinates_rest.std().item(), sep='   ')
                    CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(), sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')
                t0 = time.time()
                
            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")
            
            if iteration >= num_iterations:
                break
            
            if do_sh_warmup and (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                current_sh_levels += 1
                CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)
            
            if do_resolution_warmup and (iteration > 0) and (current_resolution_factor > 1) and (iteration % resolution_warmup_every == 0):
                current_resolution_factor /= 2.
                nerfmodel.downscale_output_resolution(1/2)
                CONSOLE.print(f'\nCamera resolution scaled to '
                        f'{nerfmodel.training_cameras.ns_cameras.height[0].item()} x '
                        f'{nerfmodel.training_cameras.ns_cameras.width[0].item()}'
                        )
                sugar.adapt_to_cameras(nerfmodel.training_cameras)
                # TODO: resize GT images
        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    return model_path