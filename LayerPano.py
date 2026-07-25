
import os
import datetime
import time
import warnings
from random import randint
from loguru import logger
warnings.filterwarnings(action='ignore')
import imageio
from typing import Optional, Dict, List, Tuple

import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from plyfile import PlyData
from torchvision.transforms import ToPILImage, ToTensor
from scipy.spatial import cKDTree



import utils.pano_utils.Equirec2Perspec as E2P
import utils.pano_utils.multi_Perspec2Equirec as m_P2E


from arguments import GSParams, CameraParams, ModelHiddenParams
try:
    from gaussian_renderer import render
    HAS_LEGACY_RENDERER = True
except Exception:
    render = None
    HAS_LEGACY_RENDERER = False
from scene import Scene, GaussianModel, LayerGaussian

from utils.loss import l1_loss, ssim, lpips_loss
from utils.camera import load_json
from utils.labelgs_mps import infer_point_labels

from utils.depth_utils import colorize
from utils.image import psnr
from utils.paint_utils import functbl
from utils.trajectory import get_pcdGenPoses
from scene.cameras import MiniCam2

try:
    from mps_splat_backend import extract_gaussian_params_from_ply, train_with_splat_apple
    HAS_SPLAT_APPLE_BACKEND = True
except Exception:
    HAS_SPLAT_APPLE_BACKEND = False


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)



def check_cuda_memo(info="", device=0):
    print(f"================= cuda memory info {info} ==================")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    
    free_memory = total_memory - (allocated_memory + cached_memory)
    
    print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Cached memory: {cached_memory / 1024**2:.2f} MB")
    print(f"Free memory: {free_memory / 1024**2:.2f} MB")
    print(f"=====================================================\n")
    

def _fill_labels_by_nearest_neighbor(points: np.ndarray, labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None

    pts = np.asarray(points, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32).reshape(-1)

    if pts.shape[0] == 0 or lab.shape[0] != pts.shape[0]:
        return lab

    known = lab > 0
    if known.all() or not known.any():
        return lab

    known_pts = pts[known]
    known_lab = lab[known]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(known_pts)
        _, nn_idx = tree.query(pts[~known], k=1)
        lab[~known] = known_lab[np.asarray(nn_idx, dtype=np.int64)]
    except Exception:
        diff = pts[~known][:, None, :] - known_pts[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argmin(dist2, axis=1)
        lab[~known] = known_lab[np.asarray(nn_idx, dtype=np.int64)]

    return lab


def _is_nonempty_training_image(image, min_nonzero_ratio: float = 1e-5) -> bool:
    arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if arr.size == 0:
        return False
    return float(np.count_nonzero(arr)) / float(arr.size) > float(min_nonzero_ratio)



class LayerPano:
    def __init__(self, save_dir=None, backend="legacy", mps_rasterizer="cpp", quality="standard", max_points=None, downsample_ratio=0.1, training_image_size=None, layer_iterations=800, background_iterations=1000, sky_iterations=500, disable_transfer=False, no_adaptive=False, repulsion_weight=1e-4, mean_lr_scale=1.0, mode="standard", early_stop_patience=None, early_stop_min_delta=0.0, lr_plateau_patience=None, lr_plateau_factor=0.5, lr_plateau_min_lr=1e-6):
        self.init_logger()
        self.save_dir = save_dir
        self.opt = GSParams()
        self.cam = CameraParams()
        self.hyper = ModelHiddenParams()
        self.backend = backend
        self.mps_rasterizer = mps_rasterizer
        self.quality = quality
        self.max_points = max_points
        self.downsample_ratio = downsample_ratio
        self.training_image_size = training_image_size
        self.layer_iterations = int(layer_iterations)
        self.background_iterations = int(background_iterations)
        self.sky_iterations = int(sky_iterations)
        self.disable_transfer = disable_transfer
        self.no_adaptive = no_adaptive
        self.repulsion_weight = repulsion_weight
        self.mean_lr_scale = mean_lr_scale
        self.mode = mode
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.lr_plateau_patience = lr_plateau_patience
        self.lr_plateau_factor = lr_plateau_factor
        self.lr_plateau_min_lr = lr_plateau_min_lr
        
        # Device selection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        bg_color = [1, 1, 1]  #[0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.step=0
        self.is_upper_mask_aggressive = True
        
        self.data_path = os.path.join(self.save_dir, 'data')
        self.pers_path = os.path.join(self.data_path, 'perspective_imgs')        

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.pers_path, exist_ok=True)
            
    def save_img(self, x, path):
        if np.max(x) > 1:
            x = x.astype(np.uint8)
        else:
            x = (x*255).astype(np.uint8)
        image = Image.fromarray(x)
        image.save(path)
        
    def init_logger(self):
        logger.remove() 
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)

    def count_layer(self, base_dir):
        count = 0
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("layer"):
                count += 1

        return count

    def readImg(self, path):
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        return img


    def create(self, input_dir, outlier_thresh,):

        input_dir = os.path.join(input_dir,'traindata')
        n_layer = self.count_layer(input_dir)
        # n_layer = 1
        print('Layers of Pano:', n_layer)
        self.outlier_thresh = outlier_thresh
        print('Outlier Thresh', self.outlier_thresh)

        quality_iteration_scale = {
            "standard": 1.0,
            "high": 1.5,
            "ultra": 2.0,
            "test": 1.0,
        }
        iter_scale = quality_iteration_scale.get(self.quality, 1.0)
        
        prev_gaussian_params = None
        prev_gaussian_labels = None
        for layer_idx in range(n_layer):
            self.traindata = self.load_pcd_and_perspectives(input_dir, layer_idx)
            if self.quality == 'test':
                n_iterations = 500
            else:
                if layer_idx == 0:
                    n_iterations = int(3001 * iter_scale)
                else:
                    n_iterations = int(2001 * iter_scale)

            if self.backend == "splat-apple":
                if not HAS_SPLAT_APPLE_BACKEND:
                    raise RuntimeError(
                        "splat-apple MLX backend is not available. Install mlx_gs dependencies."
                    )
                outfile = self.save_ply(
                    os.path.join(self.save_dir, f'gsplat_layer{layer_idx}.ply'),
                    type='mps-splat-apple'
                )
                train_with_splat_apple(
                    self.traindata,
                    outfile,
                    num_iterations=n_iterations,
                    rasterizer=self.mps_rasterizer,
                    device=self.device,
                    adaptive=(not self.no_adaptive),
                    max_points=self.max_points,
                    downsample_ratio=self.downsample_ratio,
                    repulsion_weight=self.repulsion_weight,
                    mean_lr_scale=self.mean_lr_scale,
                    early_stop_patience=self.early_stop_patience,
                    early_stop_min_delta=self.early_stop_min_delta,
                    lr_plateau_patience=self.lr_plateau_patience,
                    lr_plateau_factor=self.lr_plateau_factor,
                    lr_plateau_min_lr=self.lr_plateau_min_lr,
                    prev_gaussian_params=(None if self.disable_transfer else prev_gaussian_params),
                    prev_gaussian_labels=(None if self.disable_transfer else prev_gaussian_labels),
                )
                if not self.disable_transfer:
                    prev_gaussian_params, prev_gaussian_labels = extract_gaussian_params_from_ply(outfile)
                else:
                    prev_gaussian_params, prev_gaussian_labels = None, None
                continue

            self.gaussians = LayerGaussian(self.opt.sh_degree, outlier_thresh=self.outlier_thresh)
            if render is None:
                raise RuntimeError(
                    "Legacy gaussian_renderer backend is not available. Build the diff-gaussian-rasterization submodule or use backend='splat-apple'."
                )
            self.scene = Scene(self.traindata, gaussians_prev, self.gaussians, self.opt)        
                        
            self.training(layer_idx, n_iterations)

            
            
            
            self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            gaussians_prev = self.gaussians.wrap_gaussian()
            outfile = self.save_ply(os.path.join(self.save_dir, f'gsplat_layer{layer_idx}.ply'))

        return outfile

    def _read_layer_instances_metadata(self, input_dir):
        meta_path = os.path.join(input_dir, "traindata", "layer_instances.json")
        if not os.path.exists(meta_path):
            return None
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[WARN] Failed to read layer_instances.json: {exc}")
            return None

    def create_layer_instances(self, input_dir, outlier_thresh, metadata_path=None, background_last=True):
        input_dir = os.path.join(input_dir, "traindata")
        self.outlier_thresh = outlier_thresh

        meta = None
        if metadata_path:
            try:
                import json
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception as exc:
                print(f"[WARN] Failed to read metadata: {exc}")
        if meta is None:
            meta = self._read_layer_instances_metadata(os.path.dirname(input_dir))

        layer_order = []
        background_idx = None
        residual_idx = None
        sky_idx = None

        if isinstance(meta, dict):
            background_idx = meta.get("background_layer_idx")
            residual_idx = meta.get("residual_layer_idx")
            sky_idx = meta.get("sky_layer_idx")
            if sky_idx is None and isinstance(meta.get("sky"), dict):
                sky_idx = meta["sky"].get("layer_idx")
            if sky_idx is not None:
                sky_idx = int(sky_idx)
            instances = meta.get("instances", [])

            for item in instances:
                try:
                    layer_idx = int(item["layer_idx"])
                    if layer_idx not in layer_order:
                        layer_order.append(layer_idx)
                except Exception:
                    continue

            if residual_idx is not None:
                residual_idx = int(residual_idx)
                if residual_idx not in layer_order:
                    layer_order.append(residual_idx)

            if background_idx is not None:
                background_idx = int(background_idx)
                if background_last:
                    layer_order = [idx for idx in layer_order if idx != background_idx]
                    if residual_idx is not None and residual_idx in layer_order:
                        layer_order = [idx for idx in layer_order if idx != residual_idx] + [residual_idx]
                    layer_order.append(background_idx)
                elif background_idx not in layer_order:
                    layer_order.append(background_idx)

        if not layer_order:
            layer_order = list(range(self.count_layer(input_dir)))

        print("Instance layers:", layer_order)

        quality_iteration_scale = {
            "standard": 1.0,
            "high": 1.5,
            "ultra": 2.0,
            "test": 1.0,
        }
        iter_scale = quality_iteration_scale.get(self.quality, 1.0)

        output_paths = []
        for layer_idx in layer_order:
            layer_started = time.perf_counter()
            self.traindata = self.load_pcd_and_perspectives(input_dir, layer_idx)

            if self.quality == "test":
                n_iterations = 200
            else:
                if sky_idx is not None and layer_idx == sky_idx:
                    n_iterations = int(self.sky_iterations * iter_scale)
                elif background_idx is not None and layer_idx == background_idx:
                    n_iterations = int(self.background_iterations * iter_scale)
                else:
                    n_iterations = int(self.layer_iterations * iter_scale)

            if self.backend == "splat-apple":
                if not HAS_SPLAT_APPLE_BACKEND:
                    raise RuntimeError(
                        "splat-apple MLX backend is not available. Install mlx_gs dependencies."
                    )
                outfile = self.save_ply(
                    os.path.join(self.save_dir, f"gsplat_layer{layer_idx}.ply"),
                    type="mps-splat-apple",
                )
                train_with_splat_apple(
                    self.traindata,
                    outfile,
                    num_iterations=n_iterations,
                    rasterizer=self.mps_rasterizer,
                    device=self.device,
                    adaptive=(not self.no_adaptive),
                    max_points=self.max_points,
                    downsample_ratio=self.downsample_ratio,
                    repulsion_weight=self.repulsion_weight,
                    mean_lr_scale=self.mean_lr_scale,
                    early_stop_patience=self.early_stop_patience,
                    early_stop_min_delta=self.early_stop_min_delta,
                    lr_plateau_patience=self.lr_plateau_patience,
                    lr_plateau_factor=self.lr_plateau_factor,
                    lr_plateau_min_lr=self.lr_plateau_min_lr,
                    prev_gaussian_params=None,
                    prev_gaussian_labels=None,
                    training_profile="layer_instances",
                )
                output_paths.append(outfile)
                print(
                    f"[timing] layer={layer_idx} iterations={n_iterations} "
                    f"resolution={self.traindata['W']}x{self.traindata['H']} "
                    f"elapsed={time.perf_counter() - layer_started:.1f}s"
                )
                continue

            raise RuntimeError("Instance layer mode requires the splat-apple MLX backend")

        return output_paths
    
    def save_ply(self, fpath=None, type='3D'):
            
        if type == '3D':
            self.gaussians.save_ply(fpath)
        elif type == 'mps-splat-apple':
            # The external backend writes the output directly in LayerPano-compatible PLY.
            pass
        else:
            if not os.path.exists(fpath):
                self.gaussians_4d.save_ply(fpath)
            else:
                self.gaussians_4d.load_ply(fpath)
        return fpath


    def render_video(self, preset, phi=0):
        
        if preset == '360':
            preset='pers2pano'
            poses, theta_list, phi_list = get_pcdGenPoses(preset, {'n_views': 80, 'phi': phi})
        else:
            poses = get_pcdGenPoses(preset)

        
        videopath = os.path.join(self.save_dir, 'results', f'{preset}_v{phi}.mp4')
        depthpath = os.path.join(self.save_dir, 'results', f'depth_{preset}_v{phi}.mp4')
        
        views = []

        
        for i in range(len(poses)):
            pose = poses[i]
            cur_cam = MiniCam2(pose, self.cam.W, self.cam.H, self.cam.fovx, self.cam.fovy)
            views.append(cur_cam)
            
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8


        iterable_render = views

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth']
            framelist.append(
                np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depth = -(depth * (depth > 0)).detach().cpu().numpy()
            dmin_local = depth.min().item()
            dmax_local = depth.max().item()
            if dmin_local < dmin:
                dmin = dmin_local
            if dmax_local > dmax:
                dmax = dmax_local
            depthlist.append(depth)


        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=10, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=10, quality=8)
        return videopath, depthpath

    def render_video(self, preset, phi=0):
        
        if preset == '360':
            preset='pers2pano'
            poses, theta_list, phi_list = get_pcdGenPoses(preset, {'n_views': 80, 'phi': phi})
        else:
            poses = get_pcdGenPoses(preset)

        
        videopath = os.path.join(self.save_dir, 'results', f'{preset}_v{phi}.mp4')
        depthpath = os.path.join(self.save_dir, 'results', f'depth_{preset}_v{phi}.mp4')
        

        views = []

        
        for i in range(len(poses)):
            pose = poses[i]
            cur_cam = MiniCam2(pose, self.cam.W, self.cam.H, self.cam.fovx, self.cam.fovy)
            views.append(cur_cam)
            
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8


        iterable_render = views

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
            frame, depth = results['render'], results['depth']
            framelist.append(
                np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
            depth = -(depth * (depth > 0)).detach().cpu().numpy()
            dmin_local = depth.min().item()
            dmax_local = depth.max().item()
            if dmin_local < dmin:
                dmin = dmin_local
            if dmax_local > dmax:
                dmax = dmax_local
            depthlist.append(depth)


        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=10, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=10, quality=8)
        return videopath, depthpath

    def training(self, layer_idx, n_iterations):
        
        if not self.scene:
            raise('Build 3D Scene First!')
        
        self.opt.iterations = n_iterations

        iterable_gauss = range(1, self.opt.iterations + 1)

        # iterable_gauss = range(1, n_iterations + 1)
        tb_writer = self.prepare_logger()
        progress_bar = tqdm(range(0, n_iterations), desc="Training progress")
        ema_loss_for_log = 0.0

        # iter_start = torch.cuda.Event(enable_timing = True)
        # iter_end = torch.cuda.Event(enable_timing = True)

        for iteration in tqdm(iterable_gauss):
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # import pdb; pdb.set_trace()
            # Render
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            
            image, mask, depth = render_pkg['render'], render_pkg['mask'], render_pkg['depth'] #[c,h,w]


            viewspace_point_tensor, visibility_filter, radii = render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii']
 

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            

            Ll1 = l1_loss(image, gt_image)
            if iteration % 1000 == 1:
                print('l1 loss:', Ll1)

            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #+ 0.5 * Ldepth
            loss.backward()

            with torch.no_grad():
                # Densification
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)

                if iteration == n_iterations:
                    progress_bar.close()

                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.update_identity_mask()
                    visibility_filter_part = radii[self.gaussians.identity_mask] > 0
                    self.gaussians.max_radii2D[visibility_filter_part] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_part], radii[self.gaussians.identity_mask][visibility_filter_part])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter_part, self.gaussians.identity_mask)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.01, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
        
        
        # self.compose_pano()

    

    def compose_pano(self):
        to_pil = ToPILImage()
        phi_all = [0, -45, 45, -80, 80]
        cam_fov90 = CameraParams(fov=90)

        pers_img = []
        F_T_P = []        
        
        for phi in phi_all:

            pers_img_tmp = []
            F_T_P_tmp = []   

            persdata, theta_list, phi_list = get_pcdGenPoses("pers2pano",{'n_views': 10, 'phi': phi })    
            n_pers = len(persdata)
            
            path = os.path.join(self.pers_path, f'pers_phi{phi}');os.makedirs(path, exist_ok=True)
            
            for i in range(n_pers):
                pose = persdata[i]
                cur_cam = MiniCam2(pose, cam_fov90.W, cam_fov90.H, cam_fov90.fovx, cam_fov90.fovy)
                render_pkg = render(cur_cam, self.gaussians, self.opt, self.background, render_only=True)
                #image:[3,H,W]
                image = render_pkg['render']   # depth[1, 512, 512]
                
                image = to_pil(image.cpu()); image.save(os.path.join(path, f'pers_{i}.jpg'))
                image = np.array(image)
                
                pers_img.append(image)
                F_T_P.append([cam_fov90.fov_deg, theta_list[i], phi_list[i]])

                pers_img_tmp.append(image)
                F_T_P_tmp.append([cam_fov90.fov_deg, theta_list[i], phi_list[i]])

            ee = m_P2E.Perspective(pers_img, F_T_P)
            pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)            
            self.save_img(pano_img, os.path.join(path, f'pano_{phi}.jpg'))


            ee = m_P2E.Perspective(pers_img_tmp, F_T_P_tmp)
            pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)            
            self.save_img(pano_img, os.path.join(path, f'pano_tmp_{phi}.jpg'))

        ee = m_P2E.Perspective(pers_img, F_T_P)
        pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)
        
        self.save_img(pano_img, os.path.join(self.save_dir, f'pano.jpg'))
        self.save_img(pano_mask, os.path.join(self.save_dir, f'pano_mask.jpg'))

    def getmask(self, img):
        # img [h,w,3]
        mask = np.sum(img, axis=-1)
        mask = np.array((mask > 0)).astype(np.float32)
        return mask

    def pano2pers(self, pano, viewangle, N, time=None, name=None):
        pers_img=[]
        if not name:
            name = 'pers_split'
        equ = E2P.Equirectangular(pano)
        for i in range(N):
            theta = 360 - (viewangle/N)*i
            img = equ.GetPerspective(self.cam.fov_deg, theta, 0, self.cam.H, self.cam.W)
            
            img = np.clip(img, 0, 255).astype(np.uint8)
            if time:
                pil_img = Image.fromarray(img); pil_img.save(os.path.join(self.save_dir, f'{name}{time}_{i}.jpg'))
            else:
                pil_img = Image.fromarray(img); pil_img.save(os.path.join(self.save_dir, f'{name}_{i}.jpg'))
            pers_img.append(np.array(pil_img))
        return pers_img


    
    def load_pcd_and_perspectives(self, parent_dir, idx):
        load_dir = os.path.join(parent_dir, f'layer{idx}')
        pcd_points, pcd_colors, pcd_labels = self.load_pcd(os.path.join(load_dir, f'pcd_rgb_layer{idx}.ply'))
        _, pcd_masks, _ = self.load_pcd(os.path.join(load_dir, f'pcd_mask_layer{idx}.ply'))
        pcd_colors = pcd_colors.astype(np.float32)
        pcd_masks = pcd_masks.astype(np.float32)

        cmax = float(np.max(pcd_colors)) if pcd_colors.size else 0.0
        mmax = float(np.max(pcd_masks)) if pcd_masks.size else 0.0

        if cmax > 1.0:
            pcd_colors = pcd_colors / max(cmax, 1e-6)
        if mmax > 1.0:
            pcd_masks = pcd_masks / max(mmax, 1e-6)
        if pcd_labels is None:
            from utils.labelgs_instance_bridge import load_instance_labels_for_layer
            pcd_labels = load_instance_labels_for_layer(self.save_dir, idx, pcd_points, pcd_masks)
            if pcd_labels is None:
                pcd_labels = infer_point_labels({"pcd_points": pcd_points, "pcd_masks": pcd_masks})

        pcd_labels = _fill_labels_by_nearest_neighbor(pcd_points, pcd_labels)


        assert pcd_points.shape[0] == pcd_masks.shape[0]
        pretrain_cap = int(self.max_points) if self.max_points is not None and int(self.max_points) > 0 else None
        if pretrain_cap is not None:
            pretrain_cap = max(pretrain_cap, 2500000)
            if len(pcd_points) > pretrain_cap:
                ratio = len(pcd_points) // pretrain_cap + 1
                print('Warning: PointCloud is too large {}, downsampling by ratio of {}'.format(len(pcd_points),ratio))
                pcd_points = pcd_points[::ratio]
                pcd_colors = pcd_colors[::ratio]
                pcd_masks = pcd_masks[::ratio]
                pcd_labels = pcd_labels[::ratio]

        if 0.0 < self.downsample_ratio < 1.0 and len(pcd_points) > 0:
            target = max(1, int(round(len(pcd_points) * self.downsample_ratio)))
            if target < len(pcd_points):
                select = np.linspace(0, len(pcd_points) - 1, num=target, dtype=np.int64)
                pcd_points = pcd_points[select]
                pcd_colors = pcd_colors[select]
                pcd_masks = pcd_masks[select]
                pcd_labels = pcd_labels[select]
                print(f"[INFO] Layer {idx}: pre-downsampled points to {len(pcd_points)} ({self.downsample_ratio:.3f}x)")

        print('[INFO] !!! Loaded {} points from Layer {}.'.format(pcd_points.shape, idx))
        frames = []
        frames_dir = os.path.join(load_dir, 'frames')
        frame_paths = sorted(
            [p for p in os.listdir(frames_dir) if p.startswith('rgb_') and p.endswith('.png')]
        ) if os.path.isdir(frames_dir) else []

        def _numeric_suffix(name: str) -> int:
            digits = ''
            base = os.path.splitext(name)[0]
            for ch in reversed(base):
                if ch.isdigit():
                    digits = ch + digits
                else:
                    break
            return int(digits) if digits else -1

        frame_paths = sorted(frame_paths, key=_numeric_suffix)

        if not frame_paths:
            # Fallback to fixed 24 frames if directory listing fails
            for frame_idx in range(24):
                frame_paths.append(f'rgb_{frame_idx}.png')

        for fname in frame_paths:
            frame_idx = _numeric_suffix(fname)
            rgb_path = os.path.join(frames_dir, fname)
            pose_path = os.path.join(frames_dir, f'transform_matrix_{frame_idx}.npy')
            if not os.path.exists(rgb_path) or not os.path.exists(pose_path):
                continue
            with Image.open(rgb_path) as image_handle:
                pers_rgb = image_handle.convert("RGB").copy()
            mask_path = os.path.join(frames_dir, f'mask_{frame_idx}.png')
            if os.path.exists(mask_path):
                with Image.open(mask_path) as mask_handle:
                    supervision_mask = mask_handle.convert("L").copy()
            else:
                # Legacy traindata stored black outside each layer and had no
                # explicit mask. Infer one so old scenes no longer supervise
                # those black pixels during a retrain.
                legacy_rgb = np.asarray(pers_rgb, dtype=np.uint8)
                supervision_mask = Image.fromarray(
                    (np.any(legacy_rgb > 0, axis=-1).astype(np.uint8) * 255),
                    mode="L",
                )
            if self.training_image_size is not None and int(self.training_image_size) > 0:
                max_side = max(pers_rgb.size)
                if max_side > int(self.training_image_size):
                    resize_scale = float(self.training_image_size) / float(max_side)
                    resized = (
                        max(1, int(round(pers_rgb.size[0] * resize_scale))),
                        max(1, int(round(pers_rgb.size[1] * resize_scale))),
                    )
                    pers_rgb = pers_rgb.resize(resized, Image.Resampling.LANCZOS)
                    supervision_mask = supervision_mask.resize(
                        resized, Image.Resampling.NEAREST
                    )
            if not np.asarray(supervision_mask, dtype=np.uint8).any():
                print(f"[INFO] Layer {idx}: skipping empty frame {fname}")
                continue
            pose_gs = np.load(pose_path)
            frames.append({
                'image': pers_rgb,
                'mask': supervision_mask,
                'transform_matrix': pose_gs,
            })

        if not frames:
            raise RuntimeError(f'No non-empty frames found for layer {idx} in {frames_dir}')
        
        W, H = frames[-1]['image'].size
        erp_height = None
        erp_mask_path = os.path.join(load_dir, f"layer{idx}_erp_mask.png")
        if os.path.exists(erp_mask_path):
            with Image.open(erp_mask_path) as erp_mask_image:
                erp_height = int(erp_mask_image.height)

        self.cam.W = W
        self.cam.H = H
        self.cam.fovx = math.radians(90)
        self.cam.fovy = self.cam.H * self.cam.fovx / self.cam.W

        self.cam.fov = (self.cam.fovx, self.cam.fovy)
        self.cam.fov_deg = 90

        return {
            'fov': self.cam.fov_deg,
            'W': self.cam.W,
            'H': self.cam.H,
            'erp_height': erp_height,
            'pcd_points': pcd_points,
            'pcd_colors': pcd_colors,
            'pcd_masks': pcd_masks,
            'pcd_labels': pcd_labels,
            'frames': frames
            }

    def load_all_pcd_and_perspectives(self, parent_dir):
        layer_dirs = [
            name for name in os.listdir(parent_dir)
            if name.startswith('layer') and os.path.isdir(os.path.join(parent_dir, name))
        ]
        if not layer_dirs:
            raise RuntimeError(f'No layer directories found in {parent_dir}')

        def _layer_key(name):
            suffix = name.replace('layer', '')
            try:
                return int(suffix)
            except Exception:
                return 10**9

        layer_dirs = sorted(layer_dirs, key=_layer_key)

        all_points = []
        all_colors = []
        all_labels = []
        all_masks = []
        all_frames = []
        first_dims = None

        for layer_name in layer_dirs:
            layer_idx = _layer_key(layer_name)
            data = self.load_pcd_and_perspectives(parent_dir, layer_idx)
            all_points.append(np.asarray(data['pcd_points'], dtype=np.float32))
            all_colors.append(np.asarray(data['pcd_colors'], dtype=np.float32))
            all_masks.append(np.asarray(data['pcd_masks'], dtype=np.float32))
            all_labels.append(np.asarray(data['pcd_labels'], dtype=np.int32))
            all_frames.extend(data['frames'])
            if first_dims is None:
                first_dims = (data['W'], data['H'], data['fov'])

        merged_points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
        merged_colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.float32)
        merged_masks = np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0, 3), dtype=np.float32)
        merged_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int32)

        if first_dims is None:
            raise RuntimeError(f'Could not infer image size from {parent_dir}')

        self.cam.W, self.cam.H, self.cam.fov_deg = int(first_dims[0]), int(first_dims[1]), float(first_dims[2])
        self.cam.fovx = math.radians(90)
        self.cam.fovy = self.cam.H * self.cam.fovx / self.cam.W
        self.cam.fov = (self.cam.fovx, self.cam.fovy)

        return {
            'fov': self.cam.fov_deg,
            'W': self.cam.W,
            'H': self.cam.H,
            'pcd_points': merged_points,
            'pcd_colors': merged_colors,
            'pcd_masks': merged_masks,
            'pcd_labels': merged_labels,
            'frames': all_frames,
        }
    
    def load_pcd(self, pcd_path):
        plydata = PlyData.read(pcd_path)
        vertices = plydata['vertex']
        x, y, z = vertices['x'], vertices['y'], vertices['z']
        r, g, b = vertices['red'], vertices['green'], vertices['blue']
        points = np.stack([x, y, z], axis=-1)
        colors = np.stack([r, g, b], axis=-1)
        labels = None
        if "label" in vertices.data.dtype.names:
            labels = np.asarray(vertices['label'], dtype=np.int32)
        return points, colors, labels

    def prepare_logger(self):    
        # with open(os.path.join(self.save_dir, "cfg_args"), 'w') as cfg_log_f:
        #     cfg_log_f.write(str(Namespace(**vars(self.opt))))
        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.save_dir)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer
