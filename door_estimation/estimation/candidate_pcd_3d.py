import numpy as np
import math                     

# global img_width, img_height
# rgb_intrinsics_matrix = np.zeros((3,3)) # test_a.bag
# img_width, img_height = 1024, 768  # or 640*480 for D435i

def get_in_bbox_3d(xxmin, xxmax, yymin, yymax, downsample_rate, depth_array):
    # Approach A of matrix operation
    depth_crop = depth_array[yymin:yymax:downsample_rate,xxmin:xxmax:downsample_rate]
    
    non_zero = depth_crop[np.nonzero(depth_crop)]
    print('\n======== distance away from the detected object (mm):\n', np.mean(non_zero))

    depcol = depth_crop.flatten('F')
    depcol = np.asmatrix(depcol)
    depcol = depcol.astype(float)

    upart = np.arange(xxmin, xxmax, downsample_rate)
    ucol = np.repeat(upart,math.ceil((yymax-yymin)/downsample_rate))
    ucol = np.asmatrix(ucol)
    ucol = ucol.astype(float)
    ucxcol = ucol - cx

    vpart = np.arange(yymin, yymax, downsample_rate)
    vcol = np.tile(vpart, math.ceil((xxmax-xxmin)/downsample_rate))
    vcol = np.asmatrix(vcol)
    vcol = vcol.astype(float)
    vcycol = vcol - cy

    xcol = np.multiply(depcol, ucxcol) / fx
    ycol = np.multiply(depcol, vcycol) / fy
    # remove the empty points in the bbox
    zcol = depcol[np.nonzero(depcol)]
    xcol = xcol[np.nonzero(xcol)]
    ycol = ycol[np.nonzero(ycol)]

    in_bbox_3d = np.concatenate((xcol, ycol, zcol), axis=0)
    in_bbox_3d = np.transpose(in_bbox_3d)

    return in_bbox_3d

# the 3D points inside 2D bbox
def pcd_bbox_3d(xminl, yminl, xmaxl, ymaxl, downsample_rate, hinge_sd_pre, depth_array, class_name, items):
    quit = False
    depth_scale = 0.001 # not sure!!!!!!!!
    in_bbox_3d = []
    # rgb_intrinsics_matrix = intrinsics
    global fx, fy, cx, cy
    fx, fy, cx, cy = items[2], items[3], items[4], items[5]

    # initialize the pcd region considering the class and the hinge side
    # if class_name == 'Door':
    xxmin, xxmax, yymin, yymax = xminl, xmaxl, yminl, ymaxl
    # else: # the handle case
    #     if hinge_sd_pre == 'left': # take the region to the left of the center of handle bbox
    #         xxmin, xxmax, yymin, yymax = 0, int((xminl+xmaxl)/2), int(0.33*img_height), int(0.67*img_height)
    #     elif hinge_sd_pre == 'right': # take the region to the right of the center of handle bbox
    #         xxmin, xxmax, yymin, yymax = int((xminl+xmaxl)/2), img_width, int(0.33*img_height), int(0.67*img_height)
    #     else: # quit, instead of taking the whole region (lead to wrong normals)
    #         quit = True
    #         return in_bbox_3d, quit

    # get the 3D points inside 2D bbox
    in_bbox_3d = get_in_bbox_3d(xxmin, xxmax, yymin, yymax, downsample_rate, depth_array)
    in_bbox_3d = np.array(in_bbox_3d)
    in_bbox_3d *= depth_scale * 1000 # unit / mm
    print('\n======== Number of points inside the bounding box: \n', in_bbox_3d.shape[0])
    quit = in_bbox_3d.shape[0] < 100

    return in_bbox_3d, quit