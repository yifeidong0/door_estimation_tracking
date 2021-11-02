import pyransac3d as pyrsc
import numpy as np
from skspatial.objects import Plane, Point

# RANSAC; get the inliers 
def get_inl_ransac(in_bbox_3d, thresh, minPoints):
    quit = False
    # fit a plane
    plane1 = pyrsc.Plane()
    best_eq, idx_inliers = plane1.fit(in_bbox_3d, thresh=thresh, 
                                    minPoints=minPoints, maxIteration=50)
    if len(best_eq) == 0:
        quit = True
        inliers, door_plane, normal = 0, 0, 0
        print('\n======== SKIP THE CURRENT FRAME BECAUSE NO FIT IN RANSAC ========')
    else:
        # get the door plane for the projection of pts4
        a = -best_eq[0]/best_eq[2]
        b = -best_eq[1]/best_eq[2]
        c = -best_eq[3]/best_eq[2]
        normal = (a, b, -1)
        normal /= np.linalg.norm(normal)
        door_plane = Plane(point=[0, 0, c], normal=normal)

        print('\n======== number of inliers of the fitted plane: \n', idx_inliers.shape[0])
        print('\n======== the coefficients of the fitted plane z = ax + by + c (mm):', '\na = ', a, '\nb = ', b, '\nc = ', c)

        # get the inliers of the fitted plane
        # TODO: slow! 
        inliers = []
        for i in idx_inliers:
            inliers.append(in_bbox_3d[i])
        inliers = np.array(inliers)

    return inliers, door_plane, normal, quit

# outlier removal
def out_rmv(pcd, nb_neighbors, std_ratio):
    quit = False
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio) # ind: index
    print('\n======== number of points after outlier removal: \n', len(ind))
    if np.asarray(cl.points).shape[0] < 400:
        quit = True
        print('\n======== SKIP THE CURRENT FRAME BECAUSE NOT ENOUGH POINTS AFTER OUTLIER REMOVAL ========')

    return cl, quit 