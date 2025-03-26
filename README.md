# Parking-Lot-Management-System-

## **Overview**  
This project is a **real-time Parking Lot Detection System** that uses **YOLOv5 object detection** to identify vehicles and determine occupied and empty parking slots. The system processes a live webcam feed or a recorded video and visually marks vehicles and available slots.  

## **Features**  
**Real-time vehicle detection** using **YOLOv5**  
**Manual parking slot selection** via an interactive UI  
**Live visualization of occupied and empty parking slots**  
**Supports both webcam and video file inputs**  
 **Matplotlib-based display for compatibility in headless environments**  

## **How It Works**  
1. **Select parking slots** by clicking on the image.  
2. **YOLOv5 detects vehicles** in real time.  
3. **Each slot is analyzed** to determine whether it's occupied.  
4. **The processed video** is displayed with annotations.  

## **Installation & Setup**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/parking-lot-detection.git
cd parking-lot-detection
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Project**  
For **real-time webcam detection**:  
```bash
python main.py
```
For **video file detection**:  
```bash
python main.py --video path/to/video.mp4
```

## **Technologies Used**  
- **Python 3.12**  
- **OpenCV** (Computer Vision)  
- **YOLOv5** (Object Detection)  
- **Torch & Torchvision** (Deep Learning)  
- **Matplotlib** (Visualization)  

## **Contributing**  
Contributions are welcome! Feel free to fork, improve, and create pull requests.  
