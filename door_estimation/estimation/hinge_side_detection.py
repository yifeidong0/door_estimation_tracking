import numpy as np                       

# select from candidate detections with the highest confidence
def pick_high_conf(cls_det):
    max_det_idx = np.argmax(cls_det, axis=0)
    max_conf_det = max_det_idx[4]
    cls_hst = cls_det[max_conf_det] 

    return cls_hst

# filter out the door candidates with wrong geometric characteristics
def door_det_filter(door_det, items):
    x_thres = 0.1
    y_thres = 0.4
    xy_ratio_l = 0.30
    xy_ratio_u = 1.0

    # door width not too small
    dx_rel = (door_det[:,2] - door_det[:,0]) / items[1]
    # door height not too small
    dy_rel = (door_det[:,3] - door_det[:,1]) / items[0]
    # door width-height ratio in a proper range
    xy_ratio = (door_det[:,2] - door_det[:,0]) / (door_det[:,3] - door_det[:,1])

    logi_mat = np.stack((dx_rel>x_thres, dy_rel>y_thres, xy_ratio>xy_ratio_l, xy_ratio<xy_ratio_u), axis = 1)

    red_logi_mat = np.logical_and.reduce(logi_mat, axis = 1)
    door_det_filtered = door_det[red_logi_mat,:]

    return door_det_filtered

# hinge side detection
def hinge_det(hd_det, xcen_dr):
    if hd_det.shape[0] > 0:
        handle_hst = pick_high_conf(hd_det)
        print('\n======== handle detections of the highest confidence \n xmin, ymin, xmax, ymax, conf, cls, frame_id\n', handle_hst)
        xmin = int(handle_hst[0])
        xmax = int(handle_hst[2])
        xcen_hd = (xmin + xmax) / 2

        hinge_sd = ''
        if xcen_dr < xcen_hd:
            hinge_sd = 'left'
        else:
            hinge_sd = 'right'
        print('\n======== the estimated door hinge side: \n', hinge_sd)
    else:
        hinge_sd = 'unknown'
        print('\n======== the door hinge side cannot be estimated in this single frame! \n')

    return hinge_sd

# get the 2D bbox (door or handle) and the door hinge side
def target_2dbbox(det, expand, items):
    quit = False
    hei = items[0]
    wid = items[1]
    door_det = det[det[:,5]!=1,:]
    door_det_filtered = door_det_filter(door_det, items)
    handle_det = det[det[:,5]==1,:]

    if door_det_filtered.shape[0] > 0:
        # pick the most confident door detections
        door_hst = pick_high_conf(door_det_filtered)
        print('\n======== door detections of the highest confidence \n xmin, ymin, xmax, ymax, conf, cls, frame_id\n', door_hst)

        # door bbox
        xmin, ymin = int(door_hst[0]), int(door_hst[1])
        xmax, ymax = int(door_hst[2]), int(door_hst[3])
        xcen = (xmin+xmax) / 2

        # expand the bbox by 'expand' pixels
        xminl, yminl = max(xmin-expand, 0), max(ymin-expand, 0)
        xmaxl, ymaxl = min(xmax+expand, wid), min(ymax+expand, hei)
        class_name = 'Door'

        # hinge side detection
        hinge_sd = hinge_det(hd_det = handle_det, xcen_dr = xcen)

    elif door_det_filtered.shape[0] == 0 and handle_det.shape[0] > 0: # analyze the handle when no door detections
        hinge_sd = 'unknown'
        handle_hst = pick_high_conf(handle_det)

        # handle bbox
        xmin, ymin = int(handle_hst[0]), int(handle_hst[1])
        xmax, ymax = int(handle_hst[2]), int(handle_hst[3])
        xcen = (xmin+xmax) / 2

        # expand the bbox by 'expand' pixels
        xminl, yminl = max(xmin-4*expand, 0), max(ymin-4*expand, 0)
        xmaxl, ymaxl = min(xmax+4*expand, wid), min(ymax+4*expand, hei)
        class_name = 'Handle'

    else:
        quit = True
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!SKIP THE CURRENT FRAME BECAUSE NO HIGHLY CONFIDENT DOOR OR HANDLE DETECTIONS!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        xminl, yminl, xmaxl, ymaxl, hinge_sd, class_name = 0, 0, 0, 0, 'unknown', 'None'

    print('\n======== Pointcloud is given by: \n', class_name)

    return xminl, yminl, xmaxl, ymaxl, hinge_sd, quit, class_name