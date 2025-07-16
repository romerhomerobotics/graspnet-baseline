from segment_anything import SamPredictor, sam_model_registry
import pyrealsense2 as rs
import numpy as np
import cv2

def sam(image, prompt):
    sam = sam_model_registry["vit_l"](checkpoint="/home/kovan/beko_demo/graspnet-baseline/sam_vit_l_0b3195.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    return predictor.predict(prompt) # return masks, scores, logits


def run_realsense_sam(checkpoint_path, prompt, model_type="vit_l"):
    """
    Starts RealSense RGB stream and visualizes real-time SAM segmentation overlays.
    """
    # Initialize the SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Predict masks
            predictor.set_image(color_image)
            masks, scores, logits = predictor.predict(
                point_coords=prompt["point_coords"],
                point_labels=prompt["point_labels"],
                multimask_output=prompt.get("multimask_output", True)
            )

            # Overlay masks
            overlay = color_image.copy()
            for i, mask in enumerate(masks):
                mask = mask.astype(bool)
                color = (0, 0, 255)  # Red
                overlay[mask] = cv2.addWeighted(overlay, 0.5, np.full_like(overlay, color), 0.5, 0)[mask]

            # Show the image
            cv2.imshow("RealSense SAM Segmentation", overlay)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example prompt (this depends on how you define prompt)
    prompt = {
        "point_coords": np.array([[640, 360]]),  # Center point
        "point_labels": np.array([1]),
        "multimask_output": True
    }

    checkpoint_path = "/home/kovan/beko_demo/graspnet-baseline/sam_vit_l_0b3195.pth"
    run_realsense_sam(checkpoint_path, prompt, model_type="vit_l")
