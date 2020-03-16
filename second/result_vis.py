import json
import cv2
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

import os
import os.path
import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import pickle
import sys
from pprint import pprint
import operator
import time

from argoverse.map_representation.map_api import ArgoverseMap
from shapely.geometry.polygon import Polygon
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
am = ArgoverseMap()

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

def center_to_argo_2d_box(location,dimensions,yaw):
    crnrs = []
    for i in range(len(location)):
        x,y,z = location[i]
        l,h,w = dimensions[i]
        corners = np.array([ [l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2] ])

        rot_mat = np.array( [[ np.cos(yaw[i]) ,-np.sin(yaw[i]) ],
                            [np.sin(yaw[i]), np.cos(yaw[i])] ] )
        rot_corners = np.dot(rot_mat,corners.T).T
        rot_corners = rot_corners + [x,y]
        crnrs.append(rot_corners)
    return crnrs


track_id = {}
gt_track_id = {}
cnt,gt_cnt = 1,1
argoverse_loader = ArgoverseTrackingLoader("./../../argodataset/argoverse-tracking/val")

for seq in range(len(argoverse_loader)):
    argoverse_data = argoverse_loader[seq]
    seq_name = argoverse_loader.log_list[seq]
    print(seq_name)
    nlf = argoverse_data.num_lidar_frame
    gt_to_pred = {}
    cur_time = time.time()
    for frame in range(nlf):
        points = argoverse_data.get_lidar(frame)
        gt_objects = argoverse_data.get_label_object(frame)


        points[:,[0,1,2]] = points[:,[1,2,0]]
        points[:,[0,1]] = -points[:,[0,1]]

        bev = np.ones((5000,5000,3))
        for pt in points:
            x,z = int(np.ceil((pt[0]+250)/0.1)), int(np.ceil((pt[2]+250)/0.1))
            #if 0<=x<1200 and 0<=z<1200:
            bev[x,z] = 0


        lt = argoverse_data.lidar_timestamp_list[frame]
        path = "./../../AB3DMOT/argo_results/val_final757k_t25.0_a22_h3_ioudot10.0_ka3_v0.3_d2.0/"+seq_name +"/per_sweep_annotations_amodal/tracked_object_labels_" + str(lt) + ".json"
        
        
        print(time.time()-cur_time)
        cur_time = time.time()
        objects = json.load(open(path,'r'))
        save_corners = []

        obj_centers,obj_ids,obj_corners = [],[],[]

        tp_cnt,fp_cnt,fn_cnt,sw_cnt = 0,0,0,0
        for obj in objects:
            if obj['label_class'] == "VEHICLE":

                center = [[ obj['center']['x'], obj['center']['y'], obj['center']['z'] ]]
                dimensions = [[obj['length'],obj['height'],obj['width']]]
                quat = (obj['rotation']['w'], obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])

                yaw,pitch,roll = quaternion_to_euler(quat)
                corners = center_to_argo_2d_box(center,dimensions,[yaw])[0]
                corners = corners[:,[0,1]]
                corners[:,[1]] = -corners[:,[1]]

                if obj['track_label_uuid'] not in track_id:
                    track_id[obj['track_label_uuid']] = cnt
                    if cnt==3:
                        print(obj['track_label_uuid'])
                    cnt += 1

                obj_centers.append(center[0])
                obj_ids.append(track_id[ obj['track_label_uuid'] ] )
                obj_corners.append(corners)


        obj_centers = np.array(obj_centers)
        mark_obj = [False]*len(obj_centers)
        for gt_obj in gt_objects:
            if gt_obj.label_class == "VEHICLE":
                points = gt_obj.as_3d_bbox()
                centroid = np.reshape(np.mean(points,axis=0), (1,3) )
                test_vec = (centroid - obj_centers)
                #print(centroid.shape,obj_centers.shape,test_vec.shape)
                #print(centroid,obj_centers,test_vec)
                l2_dist = np.linalg.norm(test_vec,axis=1)
                closest_det,closest_id = np.min( l2_dist),np.argmin(l2_dist)
                #print(closest_det,closest_id,obj_ids[closest_id],centroid, test_vec[closest_id])
                #print(dir(gt_obj))

                if gt_obj.track_id not in gt_track_id:
                    gt_track_id[gt_obj.track_id] = gt_cnt
                    gt_cnt += 1



                corners = gt_obj.as_2d_bbox()
                corners = corners[:,[0,1]]
                corners[:,[1]] = -corners[:,[1]]
                corners[[2,3],:] = corners[[3,2],:]

                save_corners.append((corners,yaw))
                corners = (corners +[250,250])/0.1
                xp,yp = np.mean(corners,axis=0)
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(xp-30),int(yp-10)) 
                fontScale              = 1.2
                fontColor              = (0,0,255)
                lineType               = 8

                if closest_det<2 and not mark_obj[closest_id]:
                    if gt_obj.track_id not in gt_to_pred:
                        gt_to_pred[ gt_obj.track_id ] = obj_ids[closest_id]

                    mark_obj[closest_id] = True

                    if gt_to_pred[gt_obj.track_id] != obj_ids[closest_id]:
                        sw_cnt += 1
                        cv2.polylines(bev, np.int32([corners]), True, (255, 255, 0), 2)
                        cv2.putText(bev,f"SW", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                    else:
                        tp_cnt+=1
                        cv2.putText(bev,f"TP", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                        cv2.polylines(bev, np.int32([corners]), True, (255, 0, 0), 2)

                else:
                    fn_cnt+=1
                    cv2.putText(bev,f"FN", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
                    cv2.polylines(bev, np.int32([corners]), True, (0, 0, 255), 2)


        #if not all(mark_obj):
        #    print("yes",seq,frame)
        for fp_id,tps in enumerate(mark_obj):
            if not tps:
                fp_cnt += 1
                fp_corners = obj_corners[fp_id]
                fp_corners = (fp_corners +[250,250])/0.1
                xp,yp = np.mean(fp_corners,axis=0)
                cv2.polylines(bev, np.int32([fp_corners]), True, (0, 255, 0), 2)
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(xp-30),int(yp-10)) 
                fontScale              = 1.2
                fontColor              = (0,0,255)
                lineType               = 8

                cv2.putText(bev,f"FP", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

        fontScale=1.5
        cv2.putText(bev,f"TP:{int(tp_cnt)}",(10,100), font, fontScale,(0,0,0),lineType)
        cv2.putText(bev,f"FN:{fn_cnt}", (10,150), font, fontScale,(0,0,0),lineType)
        cv2.putText(bev,f"FP:{fp_cnt}", (10,200), font, fontScale,(0,0,0),lineType)
        cv2.putText(bev,f"SW:{sw_cnt}", (10,250), font, fontScale,(0,0,0),lineType)


        plt.figure(figsize=(20,20))
        plt.imshow(bev)
        if not os.path.exists(f"./track_analysis/{seq_name}/"):
            os.mkdir(f"./track_analysis/{seq_name}/")
        plt.savefig(f"./track_analysis/{seq_name}/pred_"+str(frame)+ ".png")
        #plt.show()
            
print(cnt,gt_cnt)    