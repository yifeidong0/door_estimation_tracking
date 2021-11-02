# optional: project corner points back to 2D
def back_proj(pts4_proj, com_3d):

    img_path = '/content/drive/MyDrive/ETH/THESIS/estimation_pip/color_aligned/color_aligned.jpeg'
    img_rgb_color = cv2.imread(img_path)
    crs_2d = []
    # u = X/Z*fx + cx
    # v = Y/Z*fy + cy
    for i in range(len(pts4_proj)):
        crs_2d.append([fx*pts4_proj[i][0]/pts4_proj[i][2]+cx, fy*pts4_proj[i][1]/pts4_proj[i][2]+cy])
    crs_2d = np.array(crs_2d)
    print('\n##### the corner points in 2D:\n', crs_2d)
    com_2d = [fx*com_3d[0]/com_3d[2]+cx, fy*com_3d[1]/com_3d[2]+cy]   
    print('\n##### the CoM point in 2D:\n', com_2d)

    # copy the rgb image
    img_rgb_cp = img_rgb_color

    # corner points in integer
    dt0 = (int(crs_2d[0][0]),int(crs_2d[0][1]))
    dt1 = (int(crs_2d[1][0]),int(crs_2d[1][1]))
    dt2 = (int(crs_2d[2][0]),int(crs_2d[2][1]))
    dt3 = (int(crs_2d[3][0]),int(crs_2d[3][1]))

    # draw circles and lines
    cc = cv2.circle(img_rgb_cp,(int(com_2d[0]),int(com_2d[1])), 5, (0,255,0), 2)
    for j in range(len(crs_2d)):
        cc = cv2.circle(img_rgb_cp,(int(crs_2d[j][0]),int(crs_2d[j][1])), 5, (0,255,0), 2)  
    cc = cv2.line(img_rgb_cp, dt0, dt1, (0,0,255), 2)
    cc = cv2.line(img_rgb_cp, dt0, dt2, (0,0,255), 2)
    cc = cv2.line(img_rgb_cp, dt3, dt1, (0,0,255), 2)
    cc = cv2.line(img_rgb_cp, dt3, dt2, (0,0,255), 2)

    # visualisation
    cv2_imshow(cc)

    return cc
