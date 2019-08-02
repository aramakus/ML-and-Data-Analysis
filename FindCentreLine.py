import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

# Read the content of simple DNA structure file
File = open("AllAtomDNA.xyz")

content = []
natoms = 0
for n, line in enumerate(File):
  if n == 0: natoms = float(line)
  if n > 1:
    tmp = str.split(line)
    for i in range(1, 4):
      tmp[i] = float(tmp[i])

    content.append(tmp)    

File.close()

# Change symbolic representations of atoms into proton number.
ChemElems = {'C' : 6.0, 'N' : 7.0, 'O' : 8.0, 'P' : 15.0 }
#for i, elem in enumerate(content): content[i][0] = ChemElems[elem[0]] 

# Sort atoms by chem elements
C = np.array([elem[1:] for elem in content if elem[0] == 'C'])
N = np.array([elem[1:] for elem in content if elem[0] == 'N'])
O = np.array([elem[1:] for elem in content if elem[0] == 'O'])
P = np.array([elem[1:] for elem in content if elem[0] == 'P'])

# TEST: keep only two points
#P = P[:3]

Px = P[:, 0]
Py = P[:, 1]
Pz = P[:, 2]

# Find boundaries
Box = []
for i in range(0, 3):
  Box.append([min(P[:, i]), max(P[:, i])])

Box = np.array(Box)
Centre = np.array([0.5*(Box[i, 1] + Box[i, 0]) for i in range(3)])

def Loss(Pts, W, B):
  '''
  line is given by the equation

  L = B + W*t

  where B and W are vectors and W*W = 1
  The latter constraint makes equations much simpler, but doesn't affect the equation.
  The distance between the point and the above line is written as

  d**2 = (B - P)**2 - [W(B - P)]**2
  '''
  distance = []

  for Pt in Pts:
    C = B - Pt
    distance.append(np.matmul(C, C) - np.matmul(C, W)**2)
  
  return np.array(distance)

def Grad(Pts, W, B, weights):
  dW = np.zeros(len(W))
  dB = np.zeros(len(B))

  for i, Pt in enumerate(Pts):
    dB = dB + weights[i]*(2*(B - Pt) - 2*W*np.matmul((B - Pt), W))
    dW = dW - weights[i]*(2*(B - Pt)*np.matmul((B - Pt), W))

  return dW, dB

# initialize random A and B
W = np.zeros(3)
B = Centre
dW = np.zeros(3)
dB = np.zeros(3)
for i in range(3):
  W[i] = np.random.uniform(low = Box[i, 0], high = Box[i, 1])
  B[i] = B[i] + np.random.uniform(low = -1, high = 1)

W = W / np.matmul(W, W)**0.5

# Find a cylinder of smallest radius that fits all the points. Use linear regression
step = 0.003
alpha = 4
for epoch in range(10):
  L = 0
  for iters in range(20):
    distances = Loss(P, W, B)
    # Loss is calculated on maximal distance.
    ind = np.argmax(distances)

    L = np.sum([elem**alpha for elem in distances])
    L = L**(1./alpha)

    dW, dB = Grad(P, W, B, distances/L)
    W -= step * dW
    B -= step * dB 

    # Scale W so that it is normalized to 1.
    W = W / np.matmul(W, W)**0.5
    
  print("epoch %d, loss %3.0f, step %1.6f" % (epoch, L, step))
  step = step / 3


print(W)
print(B)

# Determine "t" range for the line of "symmetry"
a = max(Box[0, :])
b = min(Box[1, :])
tl = max([(Box[i, 0] - B[i])/W[i] for i in range(3)])
tg = min([(Box[i, 1] - B[i])/W[i] for i in range(3)])
X = np.zeros([2, 3])
for i in range(3):
  X[0,i] = B[i] + tl*W[i]
  X[1,i] = B[i] + tg*W[i]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Px, Py, Pz, c = "orange", s = 40)

Px = C[:, 0]
Py = C[:, 1]
Pz = C[:, 2]
ax.scatter(Px, Py, Pz, c = "b", s = 10)

Px = N[:, 0]
Py = N[:, 1]
Pz = N[:, 2]
ax.scatter(Px, Py, Pz, c = "g", s = 20)

Px = O[:, 0]
Py = O[:, 1]
Pz = O[:, 2]
ax.scatter(Px, Py, Pz, c = "r", s = 20)

ax.plot(X[:, 0], X[:, 1], X[:, 2], c = 'black', lw = 4)
ax.scatter(Centre[0], Centre[1], Centre[2], c = 'pink')
plt.show()