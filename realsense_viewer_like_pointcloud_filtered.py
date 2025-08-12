import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def main():
    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create filters
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    # Create Open3D Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense Clean PointCloud", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    added = False

    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Wait for frames and align
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

            # Convert to numpy arrays
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            color_image = np.asanyarray(color_frame.get_data())

            # Mask out invalid depths: z <= 0 or z > 1.5 meters
            valid_mask = (vtx[:, 2] > 0) & (vtx[:, 2] < 1.5)
            vtx = vtx[valid_mask]
            tex = tex[valid_mask]

            # Map texture coordinates to RGB colors
            colors = []
            for u, v in tex:
                x = min(max(int(u * color_image.shape[1]), 0), color_image.shape[1] - 1)
                y = min(max(int(v * color_image.shape[0]), 0), color_image.shape[0] - 1)
                colors.append(color_image[y, x] / 255.0)

            colors = np.asarray(colors)

            # Create point cloud
            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Much stronger radius outlier removal
            pcd_clean, ind = pcd.remove_radius_outlier(
                nb_points=30,
                radius=0.05   # 5 cm radius neighborhood
            )
            print(f"Raw points: {len(vtx)}, after filtering: {len(ind)}")

            # If you want to highlight removed points, uncomment:
            # inlier_colors = np.asarray(pcd_clean.colors)
            # outlier_indices = np.setdiff1d(np.arange(len(vtx)), ind)
            # outlier_colors = np.tile([1.0, 0.0, 0.0], (len(outlier_indices), 1))
            # pcd_outliers = o3d.geometry.PointCloud()
            # pcd_outliers.points = o3d.utility.Vector3dVector(vtx[outlier_indices])
            # pcd_outliers.colors = o3d.utility.Vector3dVector(outlier_colors)

            if not added:
                vis.add_geometry(pcd_clean)
                added = True
            else:
                vis.clear_geometries()
                vis.add_geometry(pcd_clean)
                # If showing outliers too:
                # vis.add_geometry(pcd_outliers)

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vis.destroy_window()
        pipeline.stop()

if __name__ == "__main__":
    main()
