import sys
import os
sys.path.append(os.getcwd())
import cpp_engine

def run_test():
    print("Initializing MultiLayerPerceptron: 4 inputs -> 4 hidden -> 1 output.")
    mlp = cpp_engine.MultiLayerPerceptron(4, 4, 1, 0.1)

    good_inputs = cpp_engine.Matrix(1, 4, [0.5, 0.9, 0.5, 0.95])
    good_target = cpp_engine.Matrix(1, 1, [1.0])

    bad_inputs = cpp_engine.Matrix(1, 4, [0.5, 0.6, 0.5, 0.7])
    bad_target = cpp_engine.Matrix(1, 1, [0.0])

    # Check prediction BEFORE training
    pred_before = mlp.forward(good_inputs).get_data()[0]
    print(f"Prediction before training: {pred_before:.4f}")

    # Train for 2000 epochs to allow convergence
    for epoch in range(1, 2001):
        mlp.train(good_inputs, good_target)
        mlp.train(bad_inputs, bad_target)

        if epoch % 200 == 0 or epoch == 1:
            prediction_good = mlp.forward(good_inputs).get_data()[0]
            prediction_bad = mlp.forward(bad_inputs).get_data()[0]

            loss = abs(1.0 - prediction_good) + abs(0.0 - prediction_bad)
            print(f"Epoch {epoch:4d} | Total Loss: {loss:.4f} | Deep Squat Pred: {prediction_good:.4f} | Shallow Squat Pred: {prediction_bad:.4f}")

    # save weights to file
    print("Zapisywanie wyuczonych wag do pliku 'squat_weights.txt'...")
    mlp.save_weights("squat_weights.txt")
    print("Gotowe! Możesz teraz uruchomić skrypt z YOLO.")

if __name__ == "__main__":
    run_test()