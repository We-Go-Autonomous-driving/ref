import cv2
import numpy as np

def person_dist(depth_frame, cx, cy):
    box_center_roi = np.array((depth_frame[cy-10:cy+10, cx-10:cx+10]),dtype=np.float64)
    mean_dist = cv2.mean(box_center_roi)

    return mean_dist[0]

def obstacle_detect(default, depth_frame, offset = 100):
    key1 = ''
    key2 = ''

    # obstacle detection roi 10x10
    obs_roi1 = np.array((depth_frame[435:445, 195:205]),dtype=np.float64)
    obs_roi2 = np.array((depth_frame[395:405, 235:245]),dtype=np.float64)
    obs_roi3 = np.array((depth_frame[395:405, 395:405]),dtype=np.float64)
    obs_roi4 = np.array((depth_frame[435:445, 435:445]),dtype=np.float64)

    obs_roi1 = np.where(obs_roi1<1, np.mean(obs_roi1), obs_roi1)
    obs_roi2 = np.where(obs_roi2<1, np.mean(obs_roi2), obs_roi2)
    obs_roi3 = np.where(obs_roi3<1, np.mean(obs_roi3), obs_roi3)
    obs_roi4 = np.where(obs_roi4<1, np.mean(obs_roi4), obs_roi4)

    mean1 = np.mean(obs_roi1)
    mean2 = np.mean(obs_roi2)
    mean3 = np.mean(obs_roi3)
    mean4 = np.mean(obs_roi4)
    print('mean1, mean2, mean3, mean4 : ', mean1, mean2, mean3, mean4)
    
    #print('111 len(key1), len(key2) : ', len(key1), len(key2))
    #roi1_min, roi2_min, roi3_min, roi4_min = 1860, 1330, 1800, 1300
    #roi1_min, roi2_min, roi3_min, roi4_min = 1000, 1000, 1000, 1000

    roi1_min = default.roi1_th - offset
    roi2_min = default.roi2_th - offset
    roi3_min = default.roi3_th - offset
    roi4_min = default.roi4_th - offset

    #print('default distance : ', roi1_min, roi2_min, roi3_min, roi4_min)

    if mean1<roi1_min :
        print('turn right faster')
        key1 = 'right1'
        return key1
    elif mean2<roi2_min : 
        print('turn right')
        key1 = 'right2'
        return key1

    if mean4<roi4_min :
        print('turn left faster')
        key2 = 'left1'
        return key2
    elif mean3<roi3_min :
        print('turn left')
        key2 = 'left2'
        return key2
    