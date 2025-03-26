import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

class ParkingLotDetector:
    def __init__(self, video_source=0):
        """
        Initialize the parking lot detector with a video input or live webcam.
        
        Args:
            video_source (str/int): Path to the video file or 0 for webcam.
        """
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        self.parking_slots = []
        self.slot_status = []
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5
        
        print("Parking lot detection initialized!")
    
    def manual_parking_slot_selection(self):
        """Allow user to manually select parking slots using Matplotlib instead of OpenCV GUI."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error reading frame for slot selection.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Click to mark parking slots, then close the window")
        slots = []
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                slots.append((int(event.xdata), int(event.ydata)))
                plt.scatter(event.xdata, event.ydata, c='red', s=50)
                plt.draw()
        
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        self.parking_slots = slots
        print(f"Selected parking slots: {self.parking_slots}")
    
    def detect_cars(self, frame):
        """Detect cars using YOLOv5 in the given frame."""
        results = self.yolo_model(frame)
        detections = results.pandas().xyxy[0]
        
        car_boxes = []
        for _, row in detections.iterrows():
            if row['name'] in ['car', 'truck', 'bus']:  
                car_boxes.append((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
        
        return car_boxes
    
    def check_slot_occupancy(self, frame):
        """Check which parking slots are occupied based on YOLO detections."""
        car_boxes = self.detect_cars(frame)
        
        self.slot_status = []
        for (x, y) in self.parking_slots:
            occupied = any(xmin < x < xmax and ymin < y < ymax for xmin, ymin, xmax, ymax in car_boxes)
            self.slot_status.append(occupied)
        
        return self.slot_status
    
    def annotate_frame(self, frame):
        """Annotate the frame with parking slot statuses and detected vehicles."""
        car_boxes = self.detect_cars(frame)
        
        for xmin, ymin, xmax, ymax in car_boxes:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, "Vehicle", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        for i, ((x, y), occupied) in enumerate(zip(self.parking_slots, self.slot_status)):
            color = (0, 0, 255) if occupied else (0, 255, 0)
            label = "Occupied" if occupied else "Empty"
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, f"Slot {i+1}: {label}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
    
    def process_video(self):
        """Process the video or webcam feed and detect parking slot occupancy, using Matplotlib instead of OpenCV GUI."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.check_slot_occupancy(frame)
            annotated_frame = self.annotate_frame(frame)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.title("Parking Lot Detector - Press 'q' to Exit")
            plt.show(block=False)
            plt.pause(0.01)
            plt.close()
        
        self.cap.release()

if __name__ == "__main__":
    video_path = input("Enter video path (or press Enter for live webcam): ")
    video_source = video_path if video_path else 0  
    
    detector = ParkingLotDetector(video_source)
    detector.manual_parking_slot_selection()
    detector.process_video()
