import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.preprocess import _read_and_prep_v9, quaternion_to_euler
import torch

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import copy




class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, class_names, num_point_features,
                 target_assigner, feature_map_size, prep_func):
        #with open(info_path, 'rb') as f:
        #    infos = pickle.load(f)
        self._root_path = root_path
        self.argoverse_loader = ArgoverseTrackingLoader(root_path)
        self.am = ArgoverseMap()
        
        self.inform = info_path
        self._num_point_features = num_point_features
        self.class_names = class_names
        #print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        # [352, 400]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self.inform["index_list"])

    @property
    def kitti_infos(self):
        ret = []
        for info in self.inform["index_list"]:
            annos = {}
            undr_scr = info.find('_')
            seq = int(info[:undr_scr])
            frame = int(info[undr_scr+1:])
            argoverse_data = self.argoverse_loader[seq]

            ## converting to kitti camera coordinate system
            objects = argoverse_data.get_label_object(frame)

            city_name = argoverse_data.city_name
            city_to_egovehicle_se3 = argoverse_data.get_pose(frame)
            am = self.am

            gt_location = []
            dims = []
            gt_rots = []
            gt_names = []
            gt_bbox = []
            gt_occ = []
            gt_trunc = []
            gt_alpha = []

            for obj in objects:
                if obj.label_class in self.class_names:
                    corners = obj.as_3d_bbox()
                    center = np.mean(corners,axis=0)
                    center[2] = corners[2,2]
                    
                    #if np.abs(center[0]) > 32 or np.abs(center[1]) > 32 :
                    #    continue
                    
                    gt_location.append(center)
                    gt_names.append(obj.label_class)
                    quat = obj.quaternion
                    yaw,pitch,roll = quaternion_to_euler(quat)
                    if np.pi/2<= yaw <=np.pi:
                        yaw = 3*np.pi/2 -yaw
                    else:
                        yaw = -yaw-np.pi/2
                    assert -np.pi <= yaw <= np.pi

                    gt_rots.append(yaw)
                    dims.append([obj.length,obj.height,obj.width])
                    gt_bbox.append([100,100,200,200])
                    gt_occ.append(0)
                    gt_trunc.append(0)
                    gt_alpha.append(0)
                    
            gt_location,dims,gt_rots,gt_names = np.array(gt_location), np.array(dims), np.array(gt_rots), np.array(gt_names)
            gt_bbox, gt_occ, gt_trunc, gt_alpha = np.array(gt_bbox), np.array(gt_occ), np.array(gt_trunc) ,np.array(gt_alpha)
            
            roi_locations = copy.deepcopy(gt_location)
            '''
            if self.inform["include_roi"] or self.inform["dr_area"]:    
                roi_locs = city_to_egovehicle_se3.transform_point_cloud(roi_locations)  # put into city coords
                
                if self.inform["include_roi"]:
                    roi_locs_flag = am.remove_non_roi_points(roi_locs, city_name) # remove non-driveable region

                if not self.inform["include_roi"] and self.inform["dr_area"]:
                    roi_locs_flag = am.remove_non_driveable_area_points(roi_locs, city_name) # remove non-driveable region
                    
                gt_location,dims,gt_rots,gt_names = ( gt_location[roi_locs_flag] , dims[roi_locs_flag] , 
                                                     gt_rots[roi_locs_flag] , gt_names[roi_locs_flag] )

                gt_bbox, gt_occ, gt_trunc, gt_alpha = ( gt_bbox[roi_locs_flag], gt_occ[roi_locs_flag], gt_trunc[roi_locs_flag],                 gt_alpha[roi_locs_flag] )
            '''
            gt_location[:,2] -= 1.73
            gt_location[:,[0,1,2]] = gt_location[:,[1,2,0]]
            gt_location[:,[0,1]] = -gt_location[:,[0,1]]
            
            annos["name"], annos["location"], annos["dimensions"], annos["rotation_y"] = gt_names, gt_location, dims, gt_rots
            annos["bbox"], annos["alpha"], annos["occluded"], annos["truncated"] = gt_bbox, gt_alpha, gt_occ, gt_trunc
            ret.append(annos)
          
        return ret

    def __getitem__(self, idx):
        info = { "index": self.inform["index_list"][idx], "road_map": self.inform["road_map"] , "include_roadmap": self.inform["include_roadmap"], "include_road_points": self.inform["include_road_points"] ,"include_roi": self.inform["include_roi"],
              "dr_area": self.inform["dr_area"]}
        return _read_and_prep_v9(
            info=info,
            class_names=self.class_names,
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            argoverse_loader = self.argoverse_loader,
            argoverse_map = self.am,
            prep_func=self._prep_func)
