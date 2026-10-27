import os
import cv2
import csv
from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')

# define where the videos are and their labels
# 1.0 means perfect squat, 0.0 means too shallow
folders = {
    "dataset/good_squat_data": 1.0,
    "dataset/bad_squat_data": 0.0
}

# prepare csv file to save our brain food
csv_file = "squat_dataset.csv"

# open csv in write mode
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # write header
    writer.writerow(["hip_x", "hip_y", "knee_x", "knee_y", "target"])

    # loop through both folders
    for folder_path, label in folders.items():
        
        # create folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"created folder: {folder_path}. drop your videos to folder!")
            continue

        # go through every video in the folder
        for video_name in os.listdir(folder_path):
            if not video_name.endswith(('.mp4', '.mov', '.avi')):
                continue

            video_path = os.path.join(folder_path, video_name)
            print(f"extracting data from: {video_name}")

            cap = cv2.VideoCapture(video_path)
            
            max_hip_y = -1.0
            lowest_coords = [0.0, 0.0, 0.0, 0.0]

            # read video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # resize for faster processing
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                results = model(frame, verbose=False)

                # check if yolo sees a person
                if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                    keypoints = results[0].keypoints.xyn.cpu().numpy()[0]
                    
                    # make sure it sees hips and knees
                    if len(keypoints) >= 15: 
                        hip_x, hip_y = keypoints[12]
                        knee_x, knee_y = keypoints[14]

                        # find the lowest point of the squat
                        if hip_y > 0 and knee_y > 0:
                            if hip_y > max_hip_y:
                                max_hip_y = hip_y
                                lowest_coords = [hip_x, hip_y, knee_x, knee_y]

            cap.release()

            # save the best coordinates to csv if found any
            if max_hip_y > 0:
                writer.writerow([lowest_coords[0], lowest_coords[1], lowest_coords[2], lowest_coords[3], label])
                print(f"saved coordinations for: {video_name}")
            else:
                print(f"trouble with video image: {video_name}")

print("dataset is ready in squat_dataset.csv")