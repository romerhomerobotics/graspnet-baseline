import os
import sys
import numpy as np
import open3d as o3d
import argparse
import cv2
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

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
parser.add_argument('--checkpoint_path', required=True, help='GraspNet checkpoint path')
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

def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def main():
    # Load GraspNet
    net = get_net()

    # Load YOLOv8
    yolo_model = YOLO("yolov8n.pt")

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

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

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()).astype(np.float32) / 255.0
            color_image_bgr = cv2.cvtColor((color_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()

            # YOLO detection
            yolo_results = yolo_model.predict(color_image_bgr, verbose=False)

            # Gather bounding boxes
            bboxes = []
            for box in yolo_results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bboxes.append((x1, y1, x2, y2))

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

            # Point cloud
            cloud = create_point_cloud_from_depth_image(depth_image, camera, organized=True)
            mask = (depth_image > 0) & (depth_image < 1500)
            cloud_masked = cloud[mask]
            color_masked = color_image[mask]

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
            cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to("cuda:0" if torch.cuda.is_available() else "cpu")
            end_points = {'point_clouds': cloud_tensor, 'cloud_colors': color_sampled}

            # Grasp prediction
            gg = get_grasps(net, end_points)
            if cfgs.collision_thresh > 0:
                mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_masked), voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # Project grasp translations to image and filter
            keep_indices = []
            for idx, grasp in enumerate(gg):
                X, Y, Z = grasp.translation
                if Z <= 0:
                    continue
                u = intrinsics.fx * X / Z + intrinsics.ppx
                v = intrinsics.fy * Y / Z + intrinsics.ppy
                u, v = int(round(u)), int(round(v))

                in_any_box = False
                for (x1, y1, x2, y2) in bboxes:
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        in_any_box = True
                        break

                if in_any_box:
                    keep_indices.append(idx)

            # Convert back to GraspGroup
            if len(keep_indices) > 0:
                gg_filtered = gg[keep_indices]
            else:
                print("No grasps in bounding boxes. Skipping visualization.")
                continue

            if len(gg_filtered) == 0:
                print("No grasps in bounding boxes. Skipping visualization.")
                continue

            gg_filtered.nms()
            gg_filtered.sort_by_score()
            best_grasp = gg_filtered[0]
            print("Best Grasp Prediction:")
            print(" Translation (x,y,z):", best_grasp.translation)
            print(" Score:", best_grasp.score)

            gg_vis = gg_filtered[:20]

            # Open3D visualization
            pcd.points = o3d.utility.Vector3dVector(cloud_masked)
            pcd.colors = o3d.utility.Vector3dVector(color_masked)
            geometries = [pcd] + gg_vis.to_open3d_geometry_list()
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

            # OpenCV visualization with YOLO boxes
            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(color_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("RGB with YOLO Boxes", color_image_bgr)

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
