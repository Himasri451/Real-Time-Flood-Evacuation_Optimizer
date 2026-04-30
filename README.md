# 🌊 Real-Time Flood Detection And Evacuation Route Optimizer

## 🚨 Problem Statement
Floods are one of the most devastating natural disasters, causing loss of life and infrastructure damage. During such events, it becomes difficult for people to identify safe evacuation routes due to rapidly changing conditions.

Traditional systems rely on manual monitoring and static evacuation plans, which are not effective in dynamic flood scenarios. This project aims to provide a real-time AI-based solution that detects flood hazards and suggests safe evacuation routes.

---

## ⚡ Role of Edge Computing

### 📍 Components on Edge (Jetson Nano)
- YOLOv8 model for flood detection  
- CNN model for risk classification  
- A* algorithm for route optimization  
- Real-time video processing  

### ❓ Why Edge Computing
- Reduces dependency on cloud  
- Enables low-latency decision making  
- Works in low/no internet conditions  

### ✅ Benefits
- Real-time performance  
- Offline capability  
- Efficient resource usage  

---

## ⚙️ Methodology / Approach

### 🔄 Pipeline
**Input → Preprocessing → Detection → Risk Analysis → Route Optimization → Output**

### 📌 Explanation
- **Input:** Live video or recorded flood footage  
- **Preprocessing:** Frame resizing and normalization  
- **Detection:** YOLOv8 detects flood-related objects  
- **Risk Analysis:** CNN classifies zone risk (SAFE, LOW, MEDIUM, HIGH)  
- **Route Optimization:** A* finds safest evacuation path  
- **Output:** Annotated video with alerts, risk level, and route  

---

## 🤖 Model Details

- **Detection Model:** YOLOv8  
- **Classification Model:** EfficientNet-B0 (CNN)  
- **Routing Algorithm:** A* Algorithm  

### 📏 Input Size
- YOLOv8 → 640 × 640  
- CNN → 224 × 224  

### 🧠 Framework
- PyTorch  
- OpenCV  

### 🏷️ Classes Detected
- flood_water  
- submerged_vehicle  
- stranded_person  
- road_debris  
- water_marker_safe  
- water_marker_danger  
- sos_person  

---

## 🏋️ Training Details

### 📊 Dataset
- FloodNet Dataset  
- Roboflow annotated dataset  

### 🔁 Training
- YOLOv8 trained for 30 epochs  
- CNN trained for 20 epochs  
- Data augmentation applied  

### 📈 Performance
- mAP@0.5: ~0.74  
- Precision: ~0.78  
- Recall: ~0.71
-
-<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/8c9fcd0e-7825-4228-90ac-275280a1f554" />

- <img width="3000" height="2250" alt="image" src="https://github.com/user-attachments/assets/3c43fe07-6cf2-4fc0-bb39-33c6eca15bea" />


---

## 📊 Results / Output

### 🎯 System Output
- Real-time flood detection (bounding boxes)  
- Risk level display (SAFE / LOW / MEDIUM / HIGH)  
- Evacuation route with distance and ETA  
- Alert messages for high-risk zones  

### ⚡ Performance
- FPS: ~14–18  
- Inference Time: ~55–70 ms  
- Detection Confidence: ~0.82  

### 💻 Edge vs Normal
- Laptop/Desktop: ~14–18 FPS  
- Jetson Nano: ~8–12 FPS (Estimated)
- demo video
- <img width="1680" height="1050" alt="image" src="https://github.com/user-attachments/assets/bf47997d-0f70-493c-89df-f608b940a1ab" />


---

## 🛠️ Setup Instructions

### 📦 Install Dependencies
```bash
git clone <your-repo-link>
cd flood_project
pip install -r requirements.txted solution that detects flood hazards and suggests safe evacuation routes.
