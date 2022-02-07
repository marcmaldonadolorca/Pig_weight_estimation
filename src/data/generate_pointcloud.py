import numpy as np
import cv2
import glob
import open3d as o3d


image3Dnames = glob.glob('data/interim/shifted/*png')
image3Dnames.sort()
masknames = glob.glob('data/interim/masks/shifted/*png')
masknames.sort()


image3DnamesSplit = []
for i in image3Dnames:
    image3DnamesSplit.append(i.split('range')[-1])


for i in range(len(masknames)):
    if masknames[i].split('range')[-1] in image3DnamesSplit:

        # read images
        image = cv2.imread(image3Dnames[i], cv2.IMREAD_ANYDEPTH)
        mask = np.uint8(cv2.cvtColor(cv2.imread(masknames[i]), cv2.COLOR_BGR2GRAY))

        # apply masks
        image = cv2.bitwise_and(image, image, mask=mask)

        # get 3D point
        y = np.arange(0, image.shape[0], 1)
        x = np.arange(0, image.shape[1], 1)
        xx, yy = np.meshgrid(x, y)
        coords_img = np.dstack((yy, xx, image * 0.2905))
        new_coords_img = np.reshape(coords_img, (-1, 3))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_coords_img)


        points = np.asarray(pcd.points)

        # revome outlayers and not valid points
        pcd = pcd.select_by_index(np.where(points[:, 2] > 0)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < 500)[0])
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.5)

        name = masknames[i].split('range')[-1].split('.')[0]

        o3d.io.write_point_cloud('data/processed/pcd/range'+name + ".ply", pcd)


