# apply the 3D bounding box
def corners_3d(cl, door_plane, hinge_sd):
    bbox3d = o3d.geometry.OrientedBoundingBox.create_from_points(cl.points)
    com_3d = bbox3d.center
    hei_est = bbox3d.extent[0]
    wid_est = bbox3d.extent[1]

    pts8 = o3d.geometry.OrientedBoundingBox.get_box_points(bbox3d)
    pts8 = np.asarray(pts8)

    # get the 4 corners
    pts4 = [(pts8[0]+pts8[3])/2, (pts8[1]+pts8[6])/2, (pts8[2]+pts8[5])/2, (pts8[4]+pts8[7])/2]
    pts4 = np.asarray(pts4)

    # project the 4 points on the fitted plane
    # convert Ax+By+Cz+D = 0 to z = ax+by+c
    pts4_proj = []
    for i in range(len(pts4)):
        point = Point(pts4[i])
        pts4_proj.append(door_plane.project_point(point))
    pts4_proj = np.array(pts4_proj)

    # 3D output
    print('\n##### the estimated CoM position (mm): \n', com_3d)
    print('\n##### the estimated door height (mm): \n', hei_est)
    print('\n##### the estimated door width (mm): \n', wid_est)
    print('\n##### the estimated door hinge side: \n', hinge_sd)
    print('\n##### the estimated 4 corner points in 3D (mm):\n', pts4_proj, '\n############################################################################')

    return pts4_proj, com_3d, wid_est