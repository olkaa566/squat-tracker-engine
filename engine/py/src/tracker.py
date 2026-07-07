import cv2
import subprocess
from ultralytics import YOLO

print("Booting up YOLO26 AI.")
model = YOLO('yolo26m-pose.pt') 

cap = cv2.VideoCapture("assets/test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

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

    cv2.imshow("YOLO26 Squat Tracker", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("Finished")
terminal_command = [r'engine\cpp\build\Debug\engine.exe',str(lowest_hip_x), str(lowest_hip_y),str(lowest_knee_x), str(lowest_knee_y)]

subprocess.run(terminal_command, shell=True)