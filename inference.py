from ultralytics import YOLO
import cv2
import time
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def run_inference():
    print("Advanced AI Cheating Detection...")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    person_positions = {}
    head_movement_count = {}
    person_scores = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        current_persons = []
        person_id = 0

        # Detect persons
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1+x2)//2, (y1+y2)//2)

                    person_id += 1
                    current_persons.append((person_id, center))

                    if person_id not in person_positions:
                        person_positions[person_id] = center
                        head_movement_count[person_id] = 0
                        person_scores[person_id] = 0

                    # 🔥 HEAD MOVEMENT (position change)
                    old_center = person_positions[person_id]
                    movement = calculate_distance(old_center, center)

                    if movement > 20:
                        head_movement_count[person_id] += 1
                        person_scores[person_id] += 1

                    person_positions[person_id] = center

                    cv2.putText(annotated_frame,
                                f"Student {person_id}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255,255,0), 2)

        # 🔥 INTERACTION DETECTION
        for i in range(len(current_persons)):
            for j in range(i+1, len(current_persons)):
                id1, pos1 = current_persons[i]
                id2, pos2 = current_persons[j]

                dist = calculate_distance(pos1, pos2)

                if dist < 150:
                    person_scores[id1] += 2
                    person_scores[id2] += 2

                    cv2.putText(annotated_frame,
                                f"Interaction: {id1}-{id2}",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

        # 🔥 PHONE DETECTION
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "cell phone":
                    cv2.putText(annotated_frame,
                                "PHONE DETECTED",
                                (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

                    for pid in person_scores:
                        person_scores[pid] += 5

        # 🔥 FINAL STATUS
        y_offset = 120
        for pid, score in person_scores.items():

            if score > 20:
                status = "HIGH RISK"
                color = (0,0,255)
            elif score > 10:
                status = "MEDIUM"
                color = (0,165,255)
            else:
                status = "NORMAL"
                color = (0,255,0)

            text = f"Student {pid}: {status} (Score: {score})"

            cv2.putText(annotated_frame,
                        text,
                        (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            y_offset += 30

        cv2.imshow("AI Cheating Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
