import pickle
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry import Polygon
import argparse
import numpy as np
import operator
import torch
import time

# result path, range


parser = argparse.ArgumentParser(description="arg parser")



#parser.add_argument('--device', type=int, default=1, help='specify the gpu device for training')
parser.add_argument('--res_path',type=str,help='specify path where result file is stored.')
parser.add_argument('--range',type=float,help='specify range of the model for mAP evaluation')
parser.add_argument('--thresh',type=float,help='specify threshold for mAP evaluation')
parser.add_argument('--det_thresh',type=float,help='specify detection threshold for filtering detections')


args,unknown = parser.parse_known_args()

#https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
def ElevenPointInterpolatedAP(rec, prec):
    # def CalculateAveragePrecision2(rec, prec):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, 41)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 41
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]

def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]



#https://github.com/udacity/didi-competition/blob/master/tracklets/python/evaluate_tracklets.py
# 3d iou


def iou_3d(box_a, box_b,vol_a,vol_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    box_a = np.array([ box_a[:,0], box_a[:,2], box_a[:,1] ])
    box_b = np.array([ box_b[:,0], box_b[:,2], box_b[:,1] ] )
    
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    #print(xy_poly_a)
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    try:
        xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    except:
        print("ERROR",xy_poly_a,xy_poly_b)
    if xy_intersection == 0:
        return 0.
    
    
    
    vol_intersect = z_intersection * xy_intersection
    union = vol_a + vol_b - vol_intersect
    iou_val = vol_intersect/union
    
    return iou_val


def quaternion_to_euler(quat):
    w,x,y,z = quat

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw, pitch, roll


def corners_in_roi_kitti(corners):
    limit = args.range
    '''
    for i in range(4):
        if  not (-limit<= corners[i][0] <= limit and -limit <= corners[i][1] <= limit) :
            p1 = Polygon( [ (-limit,limit), (limit,limit), (limit,-limit), (-limit,-limit) ] )
            p2 = Polygon( [ (corners[0][0],corners[0][1]), (corners[1][0],corners[1][1]), (corners[2][0],corners[2][1]), (corners[3][0],corners[3][1]) ]  )

            if p1.intersection(p2).area >= 0.7*p2.area:
                return True
            else:
                return False
    '''
    for i in range(4):
        if -limit<=corners[i][0] <= limit and -limit <= corners[i][1] <= limit:
            return True
    
    return False

def center_to_bbox_2d(location,dimensions,yaw):
    crnrs = []
    for i in range(len(location)):
        x,y,z = location[i]
        l,h,w = dimensions[i]
        corners = np.array([ [l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2] ])

        rot_mat = np.array( [[ np.cos(yaw[i]) ,np.sin(yaw[i]) ],
                            [-np.sin(yaw[i]), np.cos(yaw[i])] ] )
        rot_corners = np.dot(rot_mat,corners.T).T
        rot_corners = rot_corners + [x,z]
        crnrs.append(rot_corners)
    return crnrs

def center_to_bbox_3d(location,dimensions,yaw):
    crnrs = []
    for i in range(len(location)):
        x,y,z = location[i]
        l,h,w = dimensions[i]
        corners = np.array([ [-l/2,0,w/2],[-l/2,0,-w/2],[l/2,0,-w/2],[l/2,0,w/2],
                             [-l/2,-h,w/2],[-l/2,-h,-w/2],[l/2,-h,-w/2],[l/2,-h,w/2],
                           ])

        rot_mat = np.array( [[ np.cos(yaw[i]) , 0, np.sin(yaw[i]) ],
                             [  0             , 1,     0          ], 
                             [-np.sin(yaw[i]) ,0, np.cos(yaw[i])] ]
                             )
        rot_corners = np.dot(rot_mat,corners.T).T
        rot_corners = rot_corners + [x,y,z]
        crnrs.append(rot_corners)
    return crnrs


def mAP():
    # comment corners_in_roi_kitti for normal evaluation
    # score threshold, corners_in_kitti_roi, gt_roi
    
    f = open(args.res_path,"rb")
    result = pickle.load(f)
    
    root_dir = "./../../argodataset/argoverse-tracking/val/"    
    argoverse_loader = ArgoverseTrackingLoader(root_dir)
    am = ArgoverseMap()


    preds = result
    threshold = 0.5
    out,out_3d = [],[]
    cnt = 0

    file_cnt = 0
    lcc = 0
    print(len(preds))
    for i in range(len(preds)):
        if file_cnt%50 == 0:
            print(file_cnt)
        file_cnt += 1
        if len(preds[i]['image_idx']) == 0:
            continue

        file = preds[i]['image_idx'][0]
        first_del = file.find('_')
        seq_no = int(file[:first_del])
        frame_no = int( file[ first_del + 1: ] )
        argoverse_data = argoverse_loader[seq_no]
        objects = argoverse_data.get_label_object(frame_no)
        good_result = (preds[i]['score'] > args.det_thresh)
        ps,pind,pdims = preds[i]['score'][good_result], preds[i]['name'][good_result],preds[i]['dimensions'][good_result]

        city_to_egovehicle_se3 = argoverse_data.get_pose(frame_no)
        locs,rots = preds[i]['location'][good_result], preds[i]['rotation_y'][good_result]
        city_name = argoverse_data.city_name
        '''
        for lci in range(len(locs)):
            kitti_loc = locs[lci]
            argo_loc = [kitti_loc[2],-kitti_loc[0],1.78-kitti_loc[1]]
            pts = city_to_egovehicle_se3.transform_point_cloud(np.array([argo_loc]))[0][:2]
            cl_obj,cl_conf,clines = am.get_nearest_centerline(pts,city_name) 
            if not cl_obj.is_intersection:
                vec,conf = am.get_lane_direction(pts,city_name)

                lidar_unit_vec = city_to_egovehicle_se3.inverse_transform_point_cloud(np.array([[10*vec[0],10*vec[1],15]]))[0][:2]
                lidar_orig = city_to_egovehicle_se3.inverse_transform_point_cloud( np.array([[0.,0.,15.]]) )[0][:2]
                lidar_vec = lidar_unit_vec-lidar_orig
                #print(lidar_vec)
                lidar_angle = np.arctan2(lidar_vec[1],lidar_vec[0])

                lidar_angle = -np.pi/2 - lidar_angle
                if lidar_angle > np.pi:
                    lidar_angle -= 2*np.pi
                elif lidar_angle < -np.pi:
                    lidar_angle += 2*np.pi

                if rots[lci] > np.pi:
                    rots[lci] -= 2*np.pi
                elif rots[lci] < -np.pi:
                    rots[lci] += 2*np.pi


                if conf>0.5 and np.abs(lidar_angle-rots[lci]) > 0.2 and not (np.pi-0.2 <= np.abs(lidar_angle-rots[lci]) <= np.pi+0.2 ) :
                    lcc += 1
                    #print(conf, lidar_angle, rots[lci])
                    rots[lci] = lidar_angle



        '''    

        pc = center_to_bbox_2d(preds[i]['location'][good_result],preds[i]['dimensions'][good_result],rots)
        pc_3d = center_to_bbox_3d(preds[i]['location'][good_result],preds[i]['dimensions'][good_result],rots)
        #print(preds[i]['location'])
        mark,mark_3d = {},{}
        categ = "VEHICLE"


        '''
        gt_roi = []
        gt_locss = []
        for obj in objects:
            if obj.label_class == categ:
                gt_roi.append(obj)
                corners= obj.as_3d_bbox()
                centers = np.mean(corners,axis=0)
                gt_locss.append(centers)

        roi_locs = city_to_egovehicle_se3.transform_point_cloud(np.array(gt_locss))  # put into city coords
        roi_locs_flag = am.remove_non_roi_points(roi_locs, city_name) # remove non-driveable region
        gt_roi = np.array(gt_roi)
        gt_roi = gt_roi[roi_locs_flag]
        '''            

        for obj in objects:
            if obj.label_class == categ:
                corners = obj.as_2d_bbox()
                corners = corners[:,[1,0]]
                corners[:,[0]] = -corners[:,[0]]
                corners[[2,3],:] = corners[[3,2],:]

                corners_3d = obj.as_3d_bbox()
                corners_3d = corners_3d[:,[1,2,0]]
                corners_3d[:,0] = -corners_3d[:,0]
                corners_3d[:,1] = 1.73 - corners_3d[:,1]
                corners_3d = corners_3d[[2,3,7,6,1,0,4,5],:]
                vol_gt = obj.length*obj.height*obj.width
                if corners_in_roi_kitti(corners):
                    cnt+=1
                    p1 = Polygon( [ (corners[0][0],corners[0][1]), (corners[1][0],corners[1][1]), (corners[2][0],corners[2][1]), (corners[3][0],corners[3][1]) ] )
                    mx,mx_3d = 0,0
                    temp_crnr,temp_crnr_3d = np.zeros((4,2)), np.zeros((8,3))
                    temp_scr,temp_scr_3d = 0,0
                    ind,ind_3d = -1,-1
                    
                    for num,qnt in enumerate(zip(pc,pc_3d,ps,pind,pdims)):
                        crnr,crnr_3d, score,c_ind,dims = qnt
                        pvol = dims[0]*dims[1]*dims[2]
                        if c_ind == categ:
                            p2 = Polygon( [ (crnr[0][0],crnr[0][1]), (crnr[1][0],crnr[1][1]), (crnr[2][0],crnr[2][1]), (crnr[3][0],crnr[3][1]) ]  )
                            iou = p1.intersection(p2).area / p1.union(p2).area
                            iou_td = iou_3d(corners_3d,crnr_3d,vol_gt,pvol)

                            if iou>mx :
                                mx = iou
                                temp_crnr[:] = crnr
                                temp_scr = score
                                ind = num
                                #print(iou,p1,p2,crnr_3d,corners_3d)

                            if iou_td>mx_3d :
                                mx_3d = iou_td
                                temp_crnr_3d[:] = crnr_3d
                                temp_scr_3d = score
                                ind_3d = num

                    if mx >= threshold and ind not in mark.keys() :
                        out.append((temp_scr,True))
                        mark[ind] = 1


                    if mx_3d >= threshold and ind not in mark_3d.keys() :
                        out_3d.append((temp_scr_3d,True))
                        mark_3d[ind_3d] = 1
                        #print(mx_3d)

        for sn in range(len(pind)):
            if pind[sn]==categ and sn not in mark.keys():
                out.append((ps[sn],False))
            if pind[sn] == categ and sn not in mark_3d.keys():
                out_3d.append((ps[sn],False))
        prec,prec_3d = 0,0        
        for elems in out:
            prec += elems[1]
        for elems in out_3d:
            prec_3d += elems[1]


    print("preds - ",len(out),"true predictions - ", prec , " true 3d prediction - ", prec_3d, "ground truth labels - ",cnt)    
    print(lcc)    
 
    out.sort(key = operator.itemgetter(0),reverse = True)
    out_3d.sort(key=operator.itemgetter(0),reverse=True)
    print(len(out),len(out_3d))
    
    precision, precision_3d = [], []
    recall, recall_3d = [], []

    tp = 0
    for i in range(len(out)):
        if out[i][1]:
            tp += 1
    
        precision.append(tp/(i+1))
        recall.append(tp/cnt)
    
    tp3d = 0

    for j in range(len(out_3d)):
        if out_3d[j][1]:
            tp3d += 1
    
        precision_3d.append(tp3d/(j+1))
        recall_3d.append(tp3d/cnt)
    
    '''
    plt.plot(recall,precision)
    plt.show()
    plt.plot(recall_3d,precision_3d)
    plt.show()
    '''
    
    t0 = time.time()
    area = np.max(precision)
    print(recall[0])
    #print(out[:10])
    find = 0.025
    for i in range(len(recall)):
        if recall[i] >= find:
            #print(recall[i],precision[i])
            area += np.max(precision[i:])
            find += 0.025

    
    t1 = time.time()
    
    #auth_ap_every = CalculateAveragePrecision(recall,precision)
    
    t2 = time.time()
    
    auth_ap_eleven = ElevenPointInterpolatedAP(recall,precision)
    
    t3 = time.time()
    print(t1-t0,t2-t1,t3-t2)
    
    print("Average Precision bev- ", area/41)
    #print("every point interpolation - ",auth_ap_every[0])
    print("41 point interpolation - ", auth_ap_eleven[0])

    find3d = 0.025
    area3d = np.max(precision_3d)
    
    for i in range(len(recall_3d)):
        if recall_3d[i] >= find3d:
            area3d += np.max(precision_3d[i:])
            find3d += 0.025

    #auth_ap_every_3d = CalculateAveragePrecision(recall_3d,precision_3d)
    auth_ap_eleven_3d = ElevenPointInterpolatedAP(recall_3d,precision_3d)
    
    print("Average Precision 3d- ", area3d/41)
    #print("every point interpolation - ",auth_ap_every_3d[0])
    print("41 point interpolation - ", auth_ap_eleven_3d[0])
    
    name = args.res_path.replace("pkl","txt")
    
    
    with open(name,"a+") as f:
        f.write("preds - " + str(len(out)) + " true predictions - " + str(prec) +  " true 3d prediction - " + str(prec_3d) + " ground truth labels - " + str(cnt) + "\n")    
        f.write("threshold - " + str(args.thresh) + "\n")
        
        f.write("bev AP - " +  str(area/41) + "\n")
        #f.write("every point interpolation - " + str(auth_ap_every[0]) + "\n")
        f.write("41 point interpolation - " + str(auth_ap_eleven[0]) + "\n" ) 
        
        f.write("3d AP - " + str(area3d/41) + "\n")
        #f.write("every point interpolation - " + str(auth_ap_every_3d[0]) + "\n" ) 
        f.write("41 point interpolation - " + str(auth_ap_eleven_3d[0]) + "\n" )
    
    
if __name__ == '__main__':
    mAP()

