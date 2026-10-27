import sys
import os


engine_path = r"C:\Users\ola\squat-depth-nn" 
sys.path.append(engine_path)

import cv2
from ultralytics import YOLO
import cpp_engine  

print("Booting up YOLO AI.")

#n version is faster than m one
model = YOLO('yolov8n-pose.pt') 

# initialize network
mlp = cpp_engine.MultiLayerPerceptron(4, 4, 1, 0.1)

# load the trained brain
try:
    mlp.load_weights(os.path.join(engine_path, "squat_weights.txt"))
    print("Trained weights loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load weights: {e}")

cap = cv2.VideoCapture("assets/test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# configure video saving
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_delay = int(round(1000.0 / fps)) if fps > 0 else 33

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("assets/output_video.mp4", fourcc, fps, (width // 2, height // 2))

max_hip_y = -1.0
lowest_hip_x, lowest_hip_y = 0.0, 0.0
lowest_knee_x, lowest_knee_y = 0.0, 0.0

print("Tracking started.")

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
                if hip_y > max_hip_y:
                    max_hip_y = hip_y
                    lowest_hip_x, lowest_hip_y = hip_x, hip_y
                    lowest_knee_x, lowest_knee_y = knee_x, knee_y

    annotated_frame = results[0].plot()
    
    #cv2.imshow("YOLO Squat Tracker", annotated_frame)
  
    out.write(annotated_frame)

    #if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
    #    break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking Finished. Video saved to 'assets/output_video.mp4'.")

# pass data to engine
print("Evaluating the lowest squat point in the C++ engine...")
inputs = cpp_engine.Matrix(1, 4, [float(lowest_hip_x), float(lowest_hip_y), float(lowest_knee_x), float(lowest_knee_y)])

# get the prediction
prediction = mlp.forward(inputs).get_data()[0]

print(f"C++ RESULT: {prediction:.4f}")
if prediction > 0.8:
    print("Evaluation: PERFECT SQUAT")
else:
    print("Evaluation: TOO SHALLOW")
