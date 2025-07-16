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
    # Load YOLOv8
    yolo_model = YOLO("yolov8n.pt")

    # Start RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(config)

    # Create Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense PointCloud", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    added = False

    print("Press ESC to stop.")

    try:
        # Filters
        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        pc = rs.pointcloud()

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Apply filters
            depth_frame = decimation.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Generate point cloud
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Convert RealSense frame to numpy BGR for YOLO visualization
            color_image_bgr = np.asanyarray(color_frame.get_data())

            # Convert BGR to RGB for mapping point cloud colors
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

            # Mask invalid depths
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            valid_mask = (vtx[:, 2] > 0) & (vtx[:, 2] < 1.5)
            vtx = vtx[valid_mask]
            tex = tex[valid_mask]

            # Map texture coordinates to RGB colors
            colors = []
            for u, v in tex:
                x = min(max(int(u * color_image_rgb.shape[1]), 0), color_image_rgb.shape[1] - 1)
                y = min(max(int(v * color_image_rgb.shape[0]), 0), color_image_rgb.shape[0] - 1)
                colors.append(color_image_rgb[y, x] / 255.0)
            colors = np.asarray(colors)

            # Update point cloud
            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if not added:
                vis.add_geometry(pcd)
                added = True

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # --- YOLO detection and visualization ---
            # Run YOLO on the *original BGR image* (so it looks normal in OpenCV)
            results = yolo_model.predict(color_image_bgr, verbose=False)

            # Draw detections
            vis_image = color_image_bgr.copy()
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                label = yolo_model.names[cls]

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show YOLO RGB window
            cv2.imshow("YOLOv8 RGB", vis_image)

            # Exit on ESC
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
