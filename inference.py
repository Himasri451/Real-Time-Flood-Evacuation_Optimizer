"""
inference.py - Inference Pipeline
Flood Evacuation Route Optimizer
 
Contains:
  - YOLOv8 inference with marked outputs
  - CNN inference for zone risk
  - FPS counter written on frame
  - Inference time measurement
  - Route optimizer integration
"""
import pyttsx3
import cv2
import torch
import time
import numpy as np
import heapq
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
 
from config import (
    YOLOV8_MODEL_PATH, CNN_MODEL_PATH, BASE_MODEL,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES,
    RISK_UPDATE_INTERVAL, OUTPUT_VIDEO_PATH,
    FRAME_WIDTH, FRAME_HEIGHT, START_NODE, GOAL_NODE,
    AVG_SPEED_KMPH
)
from utils import (
    Detection, ZoneStatus, EvacuationRoute,
    draw_detection, draw_zone_overlay, draw_fps,
    draw_route_overlay, assess_zone_risk, format_alert_message
)
from logger import (
    logger, DetectionLogger, AlertLogger,
    log_route_update, log_inference_stats
)
 
 
# ─────────────────────────────────────────
# ROAD NETWORK (inline for inference module)
# ─────────────────────────────────────────
 
@dataclass
class RoadNode:
    node_id: str
    name: str
    lat: float
    lon: float
 
@dataclass
class RoadEdge:
    from_node: str
    to_node: str
    distance_km: float
    road_name: str
    is_blocked: bool = False
    risk_level: str = "SAFE"
 
 
class RoadNetwork:
    def __init__(self):
        self.nodes: Dict[str, RoadNode] = {}
        self.edges: Dict[str, List[RoadEdge]] = {}
 
    def add_node(self, n: RoadNode):
        self.nodes[n.node_id] = n
        self.edges.setdefault(n.node_id, [])
 
    def add_edge(self, e: RoadEdge, bidirectional=True):
        self.edges.setdefault(e.from_node, []).append(e)
        if bidirectional:
            self.edges.setdefault(e.to_node, []).append(
                RoadEdge(e.to_node, e.from_node, e.distance_km,
                         e.road_name, e.is_blocked, e.risk_level)
            )
 
    def get_weight(self, edge: RoadEdge) -> float:
        if edge.is_blocked:
            return float('inf')
        return edge.distance_km * {"SAFE":1.0,"LOW":1.5,"MEDIUM":3.0,"HIGH":float('inf')}.get(edge.risk_level,1.0)
 
    def update_from_zone(self, zone: ZoneStatus):
        for edges in self.edges.values():
            for e in edges:
                if e.from_node == zone.zone_id:
                    e.risk_level = zone.risk_level
                    e.is_blocked = zone.risk_level == "HIGH" or (zone.flood_detected and zone.water_level == "DANGER")
 
    def astar(self, start: str, goal: str) -> Optional[EvacuationRoute]:
        if start not in self.nodes or goal not in self.nodes:
            return None
        def h(a, b):
            n1,n2 = self.nodes[a], self.nodes[b]
            R=6371; dlat=math.radians(n2.lat-n1.lat); dlon=math.radians(n2.lon-n1.lon)
            a_=math.sin(dlat/2)**2+math.cos(math.radians(n1.lat))*math.cos(math.radians(n2.lat))*math.sin(dlon/2)**2
            return R*2*math.atan2(math.sqrt(a_),math.sqrt(1-a_))
        open_set=[(0,start)]; g={start:0.0}; came_from={}; came_via={}
        while open_set:
            _,cur=heapq.heappop(open_set)
            if cur==goal:
                path,roads,warnings,node=[],[],[],cur
                while node in came_from:
                    path.append(node)
                    if node in came_via:
                        e=came_via[node]; roads.append(e.road_name)
                        if e.risk_level=="MEDIUM": warnings.append(f"Medium risk: {e.road_name}")
                    node=came_from[node]
                path.append(node); path.reverse(); roads.reverse()
                dist=round(g[goal],2)
                return EvacuationRoute(path,dist,dist,round(dist/AVG_SPEED_KMPH*60,1),roads,warnings)
            for e in self.edges.get(cur,[]):
                w=self.get_weight(e)
                if w==float('inf'): continue
                tg=g.get(cur,float('inf'))+w
                if tg<g.get(e.to_node,float('inf')):
                    came_from[e.to_node]=cur; came_via[e.to_node]=e
                    g[e.to_node]=tg; heapq.heappush(open_set,(tg+h(e.to_node,goal),e.to_node))
        return None
 
 
def build_network() -> RoadNetwork:
    net = RoadNetwork()
    for n in [
        RoadNode("A","Residential Zone A",30.90,76.85),
        RoadNode("B","Market Area B",     30.91,76.86),
        RoadNode("C","School Zone C",     30.92,76.85),
        RoadNode("D","Bridge D",          30.93,76.86),
        RoadNode("E","Highway Junction E",30.94,76.87),
        RoadNode("F","Relief Camp F",     30.96,76.88),
        RoadNode("G","Hospital Zone G",   30.92,76.88),
        RoadNode("H","Industrial Area H", 30.91,76.89),
    ]: net.add_node(n)
    for e in [
        RoadEdge("A","B",1.2,"MG Road"),
        RoadEdge("A","C",1.5,"Gandhi Street"),
        RoadEdge("B","D",2.0,"Bridge Road"),
        RoadEdge("C","D",1.8,"School Road"),
        RoadEdge("D","E",1.0,"Highway NH-5"),
        RoadEdge("E","F",2.5,"Relief Road"),
        RoadEdge("B","G",2.2,"Hospital Avenue"),
        RoadEdge("G","H",1.5,"Industrial Road"),
        RoadEdge("H","F",3.0,"Bypass Road"),
        RoadEdge("C","G",1.8,"Cross Road"),
        RoadEdge("E","H",1.2,"Link Road"),
    ]: net.add_edge(e)
    return net
 
 
# ─────────────────────────────────────────
# MAIN INFERENCE PIPELINE
# ─────────────────────────────────────────
 
class FloodInferencePipeline:
    """
    Full inference pipeline:
      1. Load YOLOv8 + CNN models
      2. For each frame: preprocess → YOLOv8 detect → CNN classify →
         route update → annotate outputs with marks, FPS, inference time
    """
 
    def __init__(self):
        self.yolo_model   = None
        self.cnn_model    = None
        self.network      = build_network()
        self.engine = pyttsx3.init()
        self.det_logger   = DetectionLogger()
        self.alert_logger = AlertLogger()
        self.zone_status  = None
        self.current_route= None
        self.last_route_t = 0
        self.frame_count  = 0
        self.fps_time     = time.time()
        self.fps          = 0.0
 
    # ── Model Loading ────────────────────────────────
 
    def load_models(self):
        """Load YOLOv8 and CNN models."""
        from ultralytics import YOLO
        from training import FloodRiskCNN
 
        # Load YOLOv8
        model_path = YOLOV8_MODEL_PATH if os.path.exists(YOLOV8_MODEL_PATH) else BASE_MODEL
        self.yolo_model = YOLO(model_path)
        logger.info(f"✅ YOLOv8 loaded: {model_path}")
 
        # Load CNN
        self.cnn_model = FloodRiskCNN(pretrained=False)
        if os.path.exists(CNN_MODEL_PATH):
            self.cnn_model.load_state_dict(
                torch.load(CNN_MODEL_PATH, map_location="cpu")
            )
            logger.info(f"✅ CNN loaded: {CNN_MODEL_PATH}")
        else:
            logger.warning("⚠️ CNN weights not found. Using untrained model.")
        self.cnn_model.eval()
 
    # ── Core Inference ───────────────────────────────
 
    def run_inference(self, frame: np.ndarray,
                      camera_id: str = "CAM_01") -> Tuple[np.ndarray, List[Detection], ZoneStatus]:
        """
        Run full inference on a single frame.
 
        Returns:
          - annotated_frame: Frame with bounding boxes, labels, FPS, inference time
          - detections:      List of Detection objects
          - zone_status:     Zone risk assessment
        """
        t_start = time.time()
 
        # ── Step 1: YOLOv8 Detection ──────────────────
        results = self.yolo_model(
    frame,
    imgsz=240,   # 🔥 ADD THIS
    conf=CONFIDENCE_THRESHOLD,
    iou=IOU_THRESHOLD,
    verbose=False
)      
        
        t_inference = (time.time() - t_start) * 1000   # ms
 
        # ── Step 2: Parse Detections ──────────────────
        detections  = []
        annotated   = frame.copy()
 
        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
 
                class_name = result.names.get(cls_id, "object")
                det = Detection(
                    class_name = class_name,
                    confidence = conf,
                    bbox       = (x1, y1, x2, y2),
                    center     = ((x1+x2)//2, (y1+y2)//2),
                    camera_id  = camera_id
                )
                detections.append(det)
                annotated = draw_detection(annotated, det)
 
                # Log detection
                zone_risk = "UNKNOWN"
                if self.frame_count %10==0:
                    self.det_logger.log_detection(camera_id, class_name, conf, (x1,y1,x2,y2), zone_risk)
 
        # ── Step 3: Zone Risk Assessment ──────────────
        zone = assess_zone_risk(detections, zone_id=camera_id)
        self.zone_status = zone
        self.det_logger.log_zone_status(
            zone.zone_id, zone.risk_level,
            zone.people_stranded, zone.vehicles_stuck, zone.sos_detected
        )
 
        # ── Step 4: CNN Risk Classification ──────────
       # CNN provides a second opinion on overall zone risk
        #if self.cnn_model is not None:
         #   from preprocessing import InputDataPreparation
          #  prep   = InputDataPreparation()
           # tensor = prep.preprocess_for_cnn(frame)
            #cnn_result = self.cnn_model.predict_risk(tensor)
            # Overlay CNN result on frame
            #cv2.putText(annotated,
             #          f"CNN Risk: {cnn_result['risk_level']} ({cnn_result['confidence']:.2f})",
              #          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
 
        # ── Step 5: Route Update (every 60s) ─────────
        if time.time() - self.last_route_t > RISK_UPDATE_INTERVAL or self.current_route is None:
            self.network.update_from_zone(zone)
            self.current_route = self.network.astar(START_NODE, GOAL_NODE)
            self.last_route_t  = time.time()
 
            if self.current_route:
                log_route_update(
                    self.current_route.path,
                    self.current_route.total_distance,
                    self.current_route.estimated_time_min,
                    self.current_route.warnings
                )
                # Log alert if HIGH risk
                if zone.risk_level in ["HIGH", "MEDIUM"]:
                    node_names = [self.network.nodes[n].name
                                  for n in self.current_route.path
                                  if n in self.network.nodes]
                    msg = format_alert_message(
                        camera_id, zone.risk_level,
                        self.current_route, node_names
                    )
                    self.alert_logger.log_alert(
                        camera_id, zone.risk_level,
                        self.current_route.path, msg,
                        self.current_route.total_distance,
                        self.current_route.estimated_time_min
                    )
 
        # ── Step 6: Annotate Frame ────────────────────
 
        # Draw zone status banner
        annotated = draw_zone_overlay(annotated, zone)
        cv2.putText(annotated, f"Water: {zone.water_level}",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2)
        # 🔥 BIG ALERT BANNER
        h, w = annotated.shape[:2]
        if zone.risk_level == "HIGH":
            
            color = (0, 0, 255)
            message = "EVACUATE IMMEDIATELY"
 
        elif zone.risk_level == "MEDIUM":
            color = (0, 165, 255)
            message = "MOVE TO SAFETY"
 
        else:
            color = (0, 200, 0)
            message = "AREA SAFE"
 
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
        annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)
 
        cv2.putText(annotated, message,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)
            # 🔊 VOICE ALERT
            #self.engine.say("Evacuate immediately")
            #self.engine.runAndWait()
 
        # Draw evacuation route overlay
        if self.current_route:
            annotated = draw_route_overlay(annotated, self.current_route)
 
        # ── FPS Counter written on frame ──────────────
        self.frame_count += 1
        current_time = time.time()
 
        if current_time - self.fps_time >= 1:
            self.fps = self.frame_count / (current_time - self.fps_time)
            self.frame_count = 0
            self.fps_time = current_time
        
            log_inference_stats(self.fps, t_inference, self.frame_count)
 
        annotated = draw_fps(annotated, self.fps, t_inference)
 
        return annotated, detections, zone
 
    # ── Batch Inference on Video ─────────────────────
 
    def run_on_video(self, video_path: str, save_output: bool = True):
        """
        Run inference on a video file.
        Saves annotated output video.
        """
        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (w, h))
 
        logger.info(f"🎬 Processing video: {video_path}")
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            annotated, _, _ = self.run_inference(frame)
 
            if save_output and writer:
                writer.write(annotated)
 
            cv2.imshow("Flood Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self.alert_logger._save()
        logger.info(f"✅ Video processing done. Output: {OUTPUT_VIDEO_PATH}")
