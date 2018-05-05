import json

import matplotlib.pyplot as plt

with open('trajectory.json', 'r') as readfile:
    trajectory = json.load(readfile)

trajectoryX = []
trajectoryY = []
for coor in trajectory['trajectory']:
    coorX, coorY = coor.split(',')
    trajectoryX.append(int(coorX))
    trajectoryY.append(int(coorY))

plt.figure()
plt.scatter(trajectoryX, trajectoryY)
plt.ylim([0, trajectory['height']])
plt.xlim([0, trajectory['width']])
plt.show()
