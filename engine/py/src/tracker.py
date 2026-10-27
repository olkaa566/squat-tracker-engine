import sys
import os
import cv2
from ultralytics import YOLO
import cpp_engine

engine_path = r"C:\Users\ola\squat-depth-nn" 
sys.path.append(engine_path)

print("booting up yolo ai.")
model = YOLO('yolov8n-pose.pt') 

# initialize network
mlp = cpp_engine.MultiLayerPerceptron(4, 4, 1, 0.1)

try:
    mlp.load_weights(os.path.join(engine_path, "squat_weights.txt"))
    print("trained weights loaded successfully\n")
except Exception as e:
    print(f"warning: could not load weights: {e}\n")

cap = cv2.VideoCapture("assets/test_video.mp4")

if not cap.isOpened():
    print("error: could not open video file.")
    exit()

# squat tracking variables
rep_count = 0
in_squat = False
current_lowest_y = 0.0
best_coords = [0.0, 0.0, 0.0, 0.0]

print("processing video for numerical data...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    results = model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        keypoints = results[0].keypoints.xyn.cpu().numpy()[0]
        
        if len(keypoints) >= 15: 
            hip_x, hip_y = keypoints[12]
            knee_x, knee_y = keypoints[14]

            if hip_y > 0 and knee_y > 0:
                
                # state 1: standing, waiting to go down
                if not in_squat:
                    if hip_y < current_lowest_y or current_lowest_y == 0.0:
                        current_lowest_y = hip_y
                    
                    # hip dropped by 5% -> squat started
                    if hip_y - current_lowest_y > 0.05:
                        in_squat = True

                # state 2: squatting, find the bottom
                else:
                    if hip_y > current_lowest_y:
                        current_lowest_y = hip_y
                        best_coords = [hip_x, hip_y, knee_x, knee_y]
                    
                    # hip went up by 10% -> rep finished
                    elif current_lowest_y - hip_y > 0.1:
                        # pass coords to c++
                        inputs = cpp_engine.Matrix(1, 4, [float(c) for c in best_coords])
                        pred = mlp.forward(inputs).get_data()[0]

                        eval_text = "PERFECT" if pred > 0.8 else "SHALLOW"
                        rep_count += 1
                        
                        # output raw data for this rep
                        print(f"rep {rep_count}:")
                        print(f"coords (hx, hy, kx, ky): [{best_coords[0]:.4f}, {best_coords[1]:.4f}, {best_coords[2]:.4f}, {best_coords[3]:.4f}]")
                        print(f"c++ score: {pred:.4f} -> {eval_text}\n")

                        # reset for next rep
                        in_squat = False
                        current_lowest_y = hip_y 

cap.release()
print(f"finished processing. total reps found: {rep_count}")