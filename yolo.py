import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Use 'n' for speed

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

            # Convert RealSense frame to numpy
            color_image = np.asanyarray(color_frame.get_data())

            # Run YOLO prediction
            results = model.predict(color_image, verbose=False)

            # Draw detections
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(color_image, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show image
            cv2.imshow("YOLOv8 RealSense Detection", color_image)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
