from sklearn.linear_model import LinearRegression
import pandas
import cv2
import open3d as o3d
import math
import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle

#Read data
data = pandas.read_csv('../../data/processed/weights.csv')
path = data['Path'].tolist()
size = data['Weight'].tolist()

#Set const
pi = np.pi
sin = np.sin
cos = np.cos

def mvee(points, tol = 0.001):
    """
        Finds the ellipse equation
        (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    return A, c

'''''''''
with open('X.bin', 'rb') as file:
    X2 = pickle.load(file)
'''''''''


X = []
Y = []
t = 0 # Counter
for i in range(len(path)):
    # Filter > 6/05
    if path[i].split('-')[-2] != '05' or (path[i].split('-')[-2] == '05' and int(path[i].split('-')[-3].split('_')[-1]) > 6): #Filtramos datos posteriores al 6/05

        # Read data
        image = cv2.imread('data/interim/shifted'+path[i].split('depth')[-1], cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread('data/interim/masks/shifted'+path[i].split('depth')[-1], 0)
        pcd = o3d.io.read_point_cloud('data/processed/pcd' + path[i].split('depth')[-1].split('.')[0] + '.ply')
        ply = o3d.io.read_triangle_mesh('data/processed/ply'+path[i].split('depth')[-1].split('.')[0]+'.ply')

        # Read data
        pack = []
        image = cv2.bitwise_and(image, image, mask=mask)
        image = image.flatten()
        image = image[image != 0]

        # number of pixels
        n_pixels = len(image)
        pack.append(n_pixels)
        # max
        max_h = np.min(image)
        pack.append(max_h)
        # mean
        mean_h = np.mean(image)
        pack.append(mean_h)
        # volume
        points = np.asarray(pcd.points)
        points = random.sample(points.tolist(), math.ceil(points.shape[0]*0.1))
        points = np.array(points)
        A, centroid = mvee(points)
        U, D, V = la.svd(A)
        rx, ry, rz = 1. / np.sqrt(D)
        u, v = np.mgrid[0:2 * pi:20j, -pi / 2:pi / 2:10j]


        def ellipse(u, v):
            x = rx * cos(u) * cos(v)
            y = ry * sin(u) * cos(v)
            z = rz * sin(v)
            return x, y, z


        E = np.dstack(ellipse(u, v))
        E = np.dot(E, V) + centroid
        x, y, z = np.rollaxis(E, axis=-1)

        # Plots volume
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, cstride=1, rstride=1, alpha=0.45)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.invert_zaxis()
        fig.savefig('data/processed/elipsoid' + path[i].split('depth')[-1].split('.')[0] + '.png')

        x=x.flatten().tolist()
        y=y.flatten().tolist()
        z=z.flatten().tolist()

        points_elipsoid = []
        for l in range(len(x)):
            points_elipsoid.append([x[l], y[l], z[l]])

        points_elipsoid = np.array(points_elipsoid)
        elipse = o3d.geometry.PointCloud()
        elipse.points = o3d.utility.Vector3dVector(points_elipsoid)
        hull, _ = elipse.compute_convex_hull()
        hull.orient_triangles()
        pack.append(hull.get_volume())


        pack.append(X2[t][3])
        t = t +1

        #Area
        area = ply.get_surface_area()
        pack.append(area)


        X.append(pack)
        Y.append(size[i])


X = np.array(X)
Y = np.array(Y)


'''''''''
with open("X.bin","wb") as f:
    pickle.dump(X, f)
with open("Y.bin", "wb") as f:
    pickle.dump(Y, f)

with open('X.bin', 'rb') as file:
    X = pickle.load(file)
with open('Y.bin', 'rb') as file:
    Y = pickle.load(file)

print('reg')
'''''''''

#Normalize
X[:][0] = X[:][0]/max(X[:][0])
X[:][1] = X[:][1]/max(X[:][1])
X[:][2] = X[:][2]/max(X[:][2])
X[:][3] = X[:][3]/max(X[:][3])
X[:][4] = X[:][4]/max(X[:][4])

#Regression
model = LinearRegression()
model.fit(X, Y)
print('coefficient of determination:', model.score(X,Y))
print('slope:', model.coef_)

model2 = LinearRegression()
model2.fit(X, Y)

pred = model2.predict(X)


print('------------------------------------------------------------')
print('MAE: ',np.sum(np.abs(pred-Y))/len(Y.tolist()))


#Save weights
data = pandas.read_csv('../../data/processed/weights_predicted.csv')
data['pred_lin'] = pred
data.to_csv(r'data/processed/weights_predicted.csv', header=True)

'''''''''''''''
coefficient of determination: 0.5503322329900848
slope: [ 1.30333286e-03 -2.10565503e-02 -1.35156751e-03 -3.86770445e-06
  3.14149902e-04]
------------------------------------------------------------
MAE:  6.432493590438632
'''''''''''''''