import sys
import os
import csv

engine_path = r"C:\Users\ola\squat-depth-nn" 
sys.path.append(engine_path)

import cpp_engine
csv_file = "squat_dataset.txt"

training_data = []
target_data = []

try:
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)

        #skip header
        next(reader) 

        for row in reader:
            if len(row) == 5:
                inputs = cpp_engine.Matrix(1, 4, [float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                training_data.append(inputs)

                #target label: 1.0 for good squat, 0.0 for bad squat
                target = cpp_engine.Matrix(1, 1, [float(row[4])])
                target_data.apend(target)

except FileNotFoundError:
    print(f"erro: {csv_file} not found. Run extract_coordinates.py first")
    exit()

print(f"loaded {len(training_data)} squats for training")

mlp = cpp_engine.MultiLayerPerceptron(4, 4, 1, 0.1)

print(f"started training on real data")

#train the network on real data
mlp.fit(training_data, target_data, 2000)

#save the weights
weights_path = os.path.join(engine_path, "squat_weights.txt")
mlp.save_weights(weights_path)

print(f"done. Saved to {weights_path}")
