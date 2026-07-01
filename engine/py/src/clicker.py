import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess 

img = mpimg.imread('assets/test_squat.png')

plt.imshow(img)
plt.title("Click 3 times: 1. Hip, 2. Knee, 3. Ankle")
print("Waiting for your clicks.")

points = plt.ginput(3) 
plt.close()

terminal_command = ['.\\eng.exe']

for x, y in points:
    norm_x = x / img.shape[1]
    norm_y = y / img.shape[0]
    terminal_command.append(str(norm_x))
    terminal_command.append(str(norm_y))

subprocess.run(terminal_command)