import sys
import os
import cv2
from ultralytics import YOLO

engine_path = r"C:\Users\ola\squat-depth-nn" 
sys.path.append(engine_path)

import cpp_engine

print("booting up yolo ai.")
model = YOLO('yolov8m-pose.pt') 

# initialize network
mlp = cpp_engine.MultiLayerPerceptron(4, 4, 1, 0.1)

try:
    mlp.load_weights(os.path.join(engine_path, "squat_weights.txt"))
    print("trained weights loaded successfully\n")
except Exception as e:
    print(f"warning: could not load weights: {e}\n")

cap = cv2.VideoCapture("assets/ola_zle.mp4")

if not cap.isOpened():
    print("error: could not open video file.")
    exit()

rep_count = 0
in_squat = False
current_lowest_y = 0.0
best_coords = [0.0, 0.0, 0.0, 0.0]
smooth_kx, smooth_ky = 0.0, 0.0
alpha = 0.4 #smooth strength

print("processing video for numerical data...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    results = model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        if len(results[0].keypoints.xyn) > 0:

            boxes = results[0].boxes.xywh.cpu().numpy()
            all_keypoints = results[0].keypoints.xyn.cpu().numpy()
            all_confs = results[0].keypoints.conf.cpu().numpy() 

            best_idx = 0
            max_area = 0
            for i in range(len(boxes)):
                area = boxes[i][2] * boxes[i][3]
                if area > max_area:
                    max_area = area
                    best_idx = i
                
            keypoints = all_keypoints[best_idx]
            confs = all_confs[best_idx]

            if len(keypoints) >= 15: 
            
                left_side = confs[11] + confs[13]
                right_side = confs[12] + confs[14]

                if left_side > right_side:
                    hip_x, hip_y = keypoints[11]
                    knee_x, knee_y = keypoints[13]
                else:
                    hip_x, hip_y = keypoints[12]
                    knee_x, knee_y = keypoints[14]

                if smooth_kx == 0.0:
                    smooth_kx, smooth_ky = knee_x, knee_y
                else:
                    smooth_kx = alpha * knee_x + (1 - alpha) * smooth_kx
                    smooth_ky = alpha * knee_y + (1 - alpha) * smooth_ky

            if hip_y > 0 and knee_y > 0:

                h, w = frame.shape[:2]

                cv2.circle(frame, (int(hip_x * w), int(hip_y * h)), 8, (0, 0, 255), -1) 
                cv2.circle(frame, (int(knee_x * w), int(knee_y * h)), 8, (0, 255, 0), -1)
                
                cv2.putText(frame, f"Reps: {rep_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                cv2.putText(frame, f"Hip Y: {hip_y:.3f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Lowest: {current_lowest_y:.3f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
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

    cv2.imshow("Squat Tracker AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print(f"finished processing. total reps found: {rep_count}")