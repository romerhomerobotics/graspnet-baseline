import os
import sys
import numpy as np
import open3d as o3d
import argparse
import cv2
import torch
import pyrealsense2 as rs
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

from graspnet_realtime import get_grasps


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


def main():
    net = get_net()

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    # Filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GraspNet PointCloud", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    added = False

    print("Press ESC to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()                
            if not depth_frame or not color_frame:
                continue

            # Apply filters
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Depth and color arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()).astype(np.float32) / 255.0
            # Turn BGR to RGB
            #color_image = color_image[:, :, ::-1]

            # Intrinsics
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            camera = CameraInfo(
                width=depth_image.shape[1],
                height=depth_image.shape[0],
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.ppx,
                cy=intrinsics.ppy,
                scale=1000.0
            )

            # Create point cloud
            cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
            mask = (depth_image > 0) & (depth_image < 1500)
            cloud_masked = cloud[mask]
            color_masked = color_image[mask]

            # Convert to Open3D point cloud
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(cloud_masked)
            pcd_raw.colors = o3d.utility.Vector3dVector(color_masked)


            # Convert back to numpy
            cloud_masked = np.asarray(pcd_raw.points)
            color_masked = np.asarray(pcd_raw.colors)

            print(f"After outlier removal: {len(cloud_masked)} points")

            # Random sampling
            if len(cloud_masked) >= cfgs.num_point:
                idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]

            # Prepare tensor
            cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to("cuda:0" if torch.cuda.is_available() else "cpu")
            end_points = {'point_clouds': cloud_sampled_tensor, 'cloud_colors': color_sampled}

            # Grasp prediction
            gg = get_grasps(net, end_points)
            if cfgs.collision_thresh > 0:
                mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_masked), voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # NMS and sort
            gg.nms()
            gg.sort_by_score()

            # Print the highest-scoring grasp
            if len(gg) > 0:
                best_grasp = gg[0]
                print("Best Grasp Prediction:")
                print(" Translation (x,y,z):", best_grasp.translation)
                print(" Rotation Matrix:\n", best_grasp.rotation_matrix)
                print(" Width:", best_grasp.width)
                print(" Score:", best_grasp.score)

                # Keep top N for visualization
                gg = gg[:20]
            else:
                print("No grasps found.")
                gg = GraspGroup()

            # Open3D visualization
            pcd.points = o3d.utility.Vector3dVector(cloud_masked)
            pcd.colors = o3d.utility.Vector3dVector(color_masked)
            geometries = [pcd] + gg.to_open3d_geometry_list()
            if not added:
                for g in geometries:
                    vis.add_geometry(g)
                added = True
            else:
                vis.clear_geometries()
                for g in geometries:
                    vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            view_ctl = vis.get_view_control()
            view_ctl.set_zoom(0.4)

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vis.destroy_window()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
