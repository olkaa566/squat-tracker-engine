import cv2
import numpy as np
import subprocess

points = []

def get_clicks(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            # green for hip, red for knee
            color = (0, 255, 0) if len(points) == 1 else (0, 0, 255)
            cv2.circle(first_frame, (x, y), 6, color, -1)
            cv2.imshow("Video", first_frame)

cap = cv2.VideoCapture("assets/test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

ret, first_frame = cap.read()

if ret:
    first_frame = cv2.resize(first_frame, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Video", first_frame)
    
    cv2.setMouseCallback("Video", get_clicks)

    #wait until 2 clicks
    while len(points) < 2:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    cv2.setMouseCallback("Video", lambda *args: None)

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = np.array([[points[0]], [points[1]]], dtype=np.float32)
lk_params = dict(winSize=(45, 45), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

max_hip_y = -1.0
lowest_hip_x, lowest_hip_y = 0.0, 0.0
lowest_knee_x, lowest_knee_y = 0.0, 0.0
height, width = first_frame.shape[:2]

while True:
    ret, frame = cap.read()

    if not ret:
        break   

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        hip_x, hip_y = p1[0][0]
        knee_x, knee_y = p1[1][0]

        if hip_y > max_hip_y:
            max_hip_y = hip_y
            lowest_hip_x, lowest_hip_y = hip_x, hip_y
            lowest_knee_x, lowest_knee_y = knee_x, knee_y

        cv2.circle(frame, (int(hip_x), int(hip_y)), 6, (0, 255, 0), -1)
        cv2.circle(frame, (int(knee_x), int(knee_y)), 6, (0, 0, 255), -1)
        
        p0 = p1.reshape(-1, 1, 2)
    # ------------------------------------------

    cv2.imshow("Video", frame)
    old_gray = frame_gray.copy()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

norm_hip_x = lowest_hip_x / width
norm_hip_y = lowest_hip_y / height
norm_knee_x = lowest_knee_x / width
norm_knee_y = lowest_knee_y / height

terminal_command = [
    '.\\eng.exe',
    str(norm_hip_x), str(norm_hip_y),
    str(norm_knee_x), str(norm_knee_y)
]

subprocess.run(terminal_command)