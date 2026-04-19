from ultralytics import YOLO
import cv2

def run_inference():
    print("Starting YOLO detection...")

    # Load model
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        # Check detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                # Detect phone (cheating)
                if label == "cell phone":
                    cv2.putText(frame, "CHEATING: PHONE DETECTED",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

        # Draw YOLO boxes
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera
    cap.release()
    cv2.destroyAllWindows()
