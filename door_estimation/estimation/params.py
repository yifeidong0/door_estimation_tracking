def params(rgb_intrinsics_matrix):
    global fx, fy, cx, cy
    fx = rgb_intrinsics_matrix[0][0] # focal length
    fy = rgb_intrinsics_matrix[1][1]
    cx = rgb_intrinsics_matrix[0][2] # principal point
    cy = rgb_intrinsics_matrix[1][2]
