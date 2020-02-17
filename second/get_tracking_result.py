import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import pickle
import math
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model_path',type=str,help='specify path where result file is stored.')
parser.add_argument('--sv_dir',type=str,help='specify path to store the results.')
parser.add_argument('--set',type=str,default='val',help='test/train/val')

#parser.add_argument('--thresh',type=float,help='specify threshold for mAP evaluation')
#parser.add_argument('--det_thresh',type=float,help='specify detection threshold for filtering detections')
args = parser.parse_args()



## NOTE : if file already exists, this code will append on file
## for storing in form of world coordinates
## second entry 1 for pedestrian, 2 for car

#sv_dir = "./AB3DMOT/data/argo/car_3d_det_val_96/"

def track_format():
    f = open(args.model_path,"rb")
    result = pickle.load(f)
    print(result[0].keys())
    
    sv_dir = args.sv_dir
    
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    else:
        for fls in os.listdir(sv_dir):
            os.remove(sv_dir+fls)

    save_path= ""

    root_dir =  os.path.join('./../../argodataset/argoverse-tracking/', args.set)
    argoverse_loader = ArgoverseTrackingLoader(root_dir)

    am = ArgoverseMap()

    for res_cnt,res in enumerate(result):
        if len(res['image_idx'])==0:
            continue
        if res_cnt%100 == 0:
            print(res_cnt)
        fr_seq = res['image_idx'][0]
        undr_scr = fr_seq.find('_')
        seq = int(fr_seq[:undr_scr])
        frame = int(fr_seq[undr_scr+1:])
        argoverse_data = argoverse_loader[seq]
        seq_id = argoverse_loader.log_list[seq]
        #save_path = os.path.join("./point_pillars/second/sample_result_ab3dmot_format/",seq_id+".txt")
        save_path = os.path.join(sv_dir,seq_id+".txt")

        city_name = argoverse_data.city_name
        city_to_egovehicle_se3 = argoverse_data.get_pose(frame)

        file = ""
        if not os.path.exists(save_path):
            file = open(save_path,"w")
        else:
            file = open(save_path,"a")

        for i in range(len(res['name'])):
            r_y = res['rotation_y'][i]
            if r_y > np.pi : r_y -= 2*np.pi
            if r_y < -np.pi : r_y += 2*np.pi

            x,y,z = res['location'][i][2], -res['location'][i][0], 1.73 - res['location'][i][1]    
            #print(z)
            roi_pts = city_to_egovehicle_se3.transform_point_cloud(np.array([[x,y,z]]))  # put into city coords
            x,y,z = roi_pts[0]
            x,y,z = -y,-z,x


            file.write( str(frame) + ", 1 ," + str(res['bbox'][i][0]) + " , " + str(res['bbox'][i][1]) + " , " +  
            str(res['bbox'][i][2])  + " , " + str(res['bbox'][i][3]) +  " , " + str(res['score'][i]) +  " , " + 
            str(res['dimensions'][i][1]) +  " , " + str(res['dimensions'][i][2]) + " , " +  str(res['dimensions'][i][0]) 
            + " , " +  str(x) + " , " +  str(y) + " , " +  
            str(z) + " , " + str(r_y) 
            + " , " + str(res['alpha'][i]) + " " +  str("\n") )
        file.close()
    
    
if __name__ == '__main__':
    track_format()