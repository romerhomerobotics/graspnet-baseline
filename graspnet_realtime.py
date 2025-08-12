"""
Realtime GraspNet inference with Intel RealSense camera.
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse

import torch
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from graspnetAPI import GraspGroup

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Number of points to sample')
parser.add_argument('--num_view', type=int, default=300, help='Number of views')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision threshold')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size')
cfgs = parser.parse_args()


def get_net():
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def get_and_process_data_realsense(pipeline, align, factor_depth=1000.0):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None, None, None

    # Convert to numpy
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Print center pixel depth to sanity-check scaling
    center_y = depth_image.shape[0] // 2
    center_x = depth_image.shape[1] // 2
    depth_raw_value = depth_image[center_y, center_x]
    depth_meters = depth_raw_value / factor_depth

    print(f"Raw depth at center pixel: {depth_raw_value}")
    print(f"Depth at center pixel in meters: {depth_meters:.3f} m")

    color_image_bgr = np.asanyarray(color_frame.get_data())
    color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Get intrinsics
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    print(f"Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    camera = CameraInfo(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.ppx,
        cy=intrinsics.ppy,
        scale=factor_depth
    )

    # Generate point cloud
    cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)

    # Workspace mask: accept all valid depth
    mask = depth_image > 0
    cloud_masked = cloud[mask]
    color_masked = color_image_rgb[mask]

    # Random sampling
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Convert to torch
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled_tensor = cloud_sampled_tensor.to(device)

    # Convert to Open3D for visualization
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled_tensor
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d, color_image_bgr, depth_image, intrinsics


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg


def run_realsense_pointcloud_viewer():
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start pipeline
    pipeline.start(config)

    # Create pointcloud object
    pc = rs.pointcloud()

    print("Press Ctrl+C to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Generate point cloud
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Convert to numpy arrays
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            color_image = np.asanyarray(color_frame.get_data())

            # Map texture coordinates to RGB colors
            colors = []
            for u, v in tex:
                x = min(max(int(u * color_image.shape[1]), 0), color_image.shape[1] - 1)
                y = min(max(int(v * color_image.shape[0]), 0), color_image.shape[0] - 1)
                colors.append(color_image[y, x] / 255.0)

            colors = np.asarray(colors)
            
            # visualize depth histogram
            depth_array = np.asanyarray(depth_frame.get_data())

            # Flatten to 1D for histogram
            depth_flat = depth_array.flatten()

            # Mask out zeros if you want to focus on valid depths
            depth_nonzero = depth_flat[depth_flat > 0]

            plt.figure(figsize=(10,4))
            plt.hist(depth_nonzero, bins=100, color='blue', alpha=0.7)
            plt.title("Histogram of Raw Depth Values")
            plt.xlabel("Depth value (units: millimeters)")
            plt.ylabel("Pixel count")
            plt.grid(True)
            plt.show()


            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Visualize
            o3d.visualization.draw_geometries([pcd])

    finally:
        pipeline.stop()


def main():
    net = get_net()

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    pipeline.start(config)

    print("Press ESC to stop.")

    try:
        while True:
            result = get_and_process_data_realsense(pipeline, align)
            if result is None or result[0] is None:
                continue

            end_points, cloud, color_bgr, depth_raw, intrinsics = result

            gg = get_grasps(net, end_points)
            if cfgs.collision_thresh > 0:
                gg = collision_detection(gg, np.asarray(cloud.points))

            # Visualize RGB and depth in OpenCV
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.imshow("RGB", color_bgr)
            cv2.imshow("Depth", depth_colormap)

            # Visualize grasps in Open3D
            gg.nms()
            gg.sort_by_score()
            gg = gg[:50]
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])

            # Check for ESC key
            key = cv2.waitKey(1)
            if key == 27:
                print("Stopping...")
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def run_realsense_rgb_viewer():
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream only (you can also enable depth if you want)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start pipeline
    pipeline.start(config)

    print("Press ESC to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            print("Depth stats:")
            print(f"min: {np.min(depth_image)}, max: {np.max(depth_image)}, mean: {np.mean(depth_image)}")
            # Mask invalid depths
            valid_depth = np.where(depth_image == 65535, 0, depth_image)

            # Optionally, clip to max range (e.g., 1500 mm)
            clipped_depth = np.clip(valid_depth, 0, 1500)

            # Normalize to 0–255
            depth_normalized = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX)

            # Convert to uint8
            depth_normalized = depth_normalized.astype(np.uint8)

            # Colormap
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


            # Show image
            cv2.imshow("RealSense RGB", color_image)
            cv2.imshow("RealSense Depth", depth_colormap)

            # Exit on ESC key
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    run_realsense_pointcloud_viewer()
    # run_realsense_rgb_viewer()
    # main()