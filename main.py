
"""
main.py - Main Entry Point
Flood Evacuation Route Optimizer
 
Logic:
  a. Initialise the system
  b. Load input source
  c. Call inference pipeline
  d. Display and output
"""
import time
import cv2
import os
import sys
from config import INPUT_SOURCE, CAMERA_ID, OUTPUT_VIDEO_PATH
from preprocessing import InputDataPreparation
from inference import FloodInferencePipeline
from logger import logger, log_system_start, log_system_stop, log_model_loaded
from utils import get_video_writer
 
 
# ─────────────────────────────────────────
# a. INITIALISE THE SYSTEM
# ─────────────────────────────────────────
 
def initialise_system() -> FloodInferencePipeline:
    """
    Initialise all system components:
      - Load YOLOv8 model
      - Load CNN model
      - Set up loggers
      - Build road network
    """
    log_system_start()
    logger.info("⚙️  Initialising Flood Evacuation System...")
 
    pipeline = FloodInferencePipeline()
    pipeline.load_models()
 
    log_model_loaded("YOLOv8 + CNN FloodRiskCNN")
    logger.info("✅ System initialised successfully.")
    return pipeline
 
 
# ─────────────────────────────────────────
# b. LOAD INPUT SOURCE
# ─────────────────────────────────────────
 
def load_input_source(source=INPUT_SOURCE) -> InputDataPreparation:
    """
    Load and validate input source.
    Supports: webcam (0), video file path.
    """
    logger.info(f"📷 Loading input source: {source}")
    data_input = InputDataPreparation(source=source)
 
    if not data_input.initialize():
        logger.error("❌ Failed to open input source. Exiting.")
        sys.exit(1)
 
    logger.info("✅ Input source loaded.")
    return data_input
 
 
# ─────────────────────────────────────────
# c. CALL INFERENCE PIPELINE
# ─────────────────────────────────────────
 
def run_inference_pipeline(pipeline: FloodInferencePipeline,
                            data_input: InputDataPreparation,
                            save_output: bool = True):
    """
    Main inference loop:
      - Reads frames from input
      - Preprocesses each frame
      - Calls YOLOv8 + CNN inference
      - Updates evacuation routes
      - Returns annotated frames for display
    """
    logger.info("🚀 Inference pipeline running...")
 
    # Set up video writer for output saving
    writer = None
    first_frame_done = False
 
    try:
        frame_count = 0
        while True:
            # Read frame
            ret, frame = data_input.read_frame()
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            if not ret or frame is None:
                logger.info("📼 End of input stream.")
                break
 
            # Preprocess frame for YOLOv8
            frame = cv2.resize(frame, (1280, 720))
            processed = data_input.preprocess_for_yolov8(frame)
 
            # Run inference (detection + classification + routing)
            annotated, detections, zone = pipeline.run_inference(
                processed, camera_id=CAMERA_ID
            )
 
            # Initialize writer on first frame
            if save_output and not first_frame_done:
                h, w = annotated.shape[:2]
                writer = get_video_writer(OUTPUT_VIDEO_PATH, w, h, fps=20.0)
                first_frame_done = True
                logger.info(f"💾 Output video: {OUTPUT_VIDEO_PATH}")
 
            if writer:
                writer.write(annotated)
 
            yield annotated, detections, zone
 
    finally:
        if writer:
            writer.release()
        data_input.release()
        pipeline.alert_logger._save()
 
 
# ─────────────────────────────────────────
# d. DISPLAY AND OUTPUT
# ─────────────────────────────────────────
 
def display_output(frame_generator):
    """
    Display annotated frames in OpenCV window.
    Press 'Q' to quit.
    """
    logger.info("🖥️  Displaying output. Press 'Q' to quit.")
 
    WINDOW_NAME = "Flood Evacuation System"
 
    # ✅ FIX 1: Create window ONCE before the loop
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # ✅ FIX 2: Auto fullscreen on launch — no clicking needed
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
 
    for annotated, detections, zone in frame_generator:
        # ✅ FIX 3: Consistent window name
        cv2.imshow(WINDOW_NAME, annotated)
 
        time.sleep(0.03)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cv2.destroyAllWindows()
    log_system_stop()
    print("\n✅ System stopped.")
 
 
# ─────────────────────────────────────────
# DEMO MODE (No Camera Required)
# ─────────────────────────────────────────
 
def run_demo():
    """
    Demo mode — demonstrates route optimizer without a live camera.
    Perfect for showing evaluators the system logic.
    """
    from inference import build_network
    log_system_start()
    print("\n🌊 FLOOD EVACUATION SYSTEM — DEMO MODE")
    print("="*55)
 
    network = build_network()
 
    print("\n📍 Scenario: Major flood event in city")
    print("   Simulating YOLOv8 detection results...\n")
 
    # Simulate YOLOv8 detections causing road blockages
    from utils import ZoneStatus
    import time
 
    blocked_zone = ZoneStatus(
        zone_id="B", risk_level="HIGH",
        flood_detected=True, people_stranded=3,
        vehicles_stuck=2, water_level="DANGER",
        sos_detected=True, timestamp=time.time()
    )
    network.update_from_zone(blocked_zone)
    print(f"🔴 YOLOv8 detected: {blocked_zone.people_stranded} people stranded in Zone B")
    print(f"🚗 YOLOv8 detected: {blocked_zone.vehicles_stuck} vehicles submerged in Zone B")
    print(f"🆘 YOLOv8 detected: SOS signal in Zone B — Bridge Road BLOCKED\n")
 
    route = network.astar("A", "F")
    if route:
        print("✅ Safest evacuation route calculated by A* algorithm:")
        print(f"   Path: {' → '.join(route.path)}")
        print(f"   Roads: {' → '.join(route.road_names)}")
        print(f"   Distance: {route.total_distance} km")
        print(f"   ETA: {route.estimated_time_min} minutes")
        if route.warnings:
            for w in route.warnings:
                print(f"   ⚠️  {w}")
    else:
        print("❌ No safe route — all paths blocked!")
 
    print("\n🔄 Simulating 60-second update: Bridge Road cleared...")
    blocked_zone.risk_level  = "LOW"
    blocked_zone.flood_detected = False
    blocked_zone.water_level = "SAFE"
    network.update_from_zone(blocked_zone)
 
    route2 = network.astar("A", "F")
    if route2:
        print("✅ Updated route:")
        print(f"   Path: {' → '.join(route2.path)}")
        print(f"   Distance: {route2.total_distance} km")
 
    log_system_stop()
 
 
# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
 
if __name__ == "__main__":
    mode   = sys.argv[1] if len(sys.argv) > 1 else "demo"
    source = sys.argv[2] if len(sys.argv) > 2 else INPUT_SOURCE
 
    if mode == "demo":
        run_demo()
 
    elif mode == "run":
        # a. Initialise
        pipeline    = initialise_system()
        # b. Load input
        data_input  = load_input_source(source=source)
        # c. Inference pipeline
        frame_gen   = run_inference_pipeline(pipeline, data_input, save_output=True)
        # d. Display output
        display_output(frame_gen)
 
    elif mode == "train":
        from training import YOLOv8Trainer, FloodRiskCNN, CNNTrainer
        print("🚀 Training YOLOv8...")
        trainer = YOLOv8Trainer()
        trainer.train()
        trainer.validate()
 
    elif mode == "video":
        pipeline = initialise_system()
        pipeline.run_on_video(source, save_output=True)
 
    else:
        print("Usage:")
        print("  python main.py demo              # Demo mode (no camera)")
        print("  python main.py run               # Run with webcam")
        print("  python main.py run video.mp4     # Run with video file")
        print("  python main.py video video.mp4   # Process & save video")
        print("  python main.py train             # Train models")
