import os
import sys
import numpy as np
import open3d as o3d
import torch
import pyrealsense2 as rs
import argparse

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--num_view', type=int, default=300)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size', type=float, default=0.01)
cfgs = parser.parse_args()

# --- Load model ---
def get_net():
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01,0.02,0.03,0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("Model loaded.")
    net.device = device 
    return net, device
# --- Get one RGBD frame from RealSense ---
def get_realsense_frame():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    for _ in range(10):  # warm-up
        frames = pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    pipeline.stop()

    if not depth_frame or not color_frame:
        raise RuntimeError("Failed to get frames from camera.")

    # Convert to numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()) / 255.0  # Normalize to 0–1
    color_image = color_image.astype(np.float32)

    # Intrinsics
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    camera = CameraInfo(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
        scale=1000.0  # because depth is in mm
    )
    return color_image, depth_image, camera

# --- Preprocess data like in demo ---
def prepare_graspnet_input(color, depth, camera):
    # Use simple workspace mask: depth > 0
    mask = depth > 0
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # Sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Format input
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(net.device)
    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_masked, color_masked

# --- Run grasp prediction ---
def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

# --- Collision filtering ---
def collision_detection(gg, cloud_np):
    mfcdetector = ModelFreeCollisionDetector(cloud_np, voxel_size=cfgs.voxel_size)
    mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    return gg[~mask]

# --- Visualize ---
def visualize(gg, cloud_masked, color_masked):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

# --- Run full demo ---
if __name__ == '__main__':
    net, device = get_net()
    color, depth, cam_info = get_realsense_frame()
    end_points, cloud_masked, color_masked = prepare_graspnet_input(color, depth, cam_info)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, cloud_masked)
    visualize(gg, cloud_masked, color_masked)
