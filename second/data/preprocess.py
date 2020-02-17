import pathlib
import pickle
import time
from collections import defaultdict

import numpy as np
from skimage import io as imgio

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.core.point_cloud.bev_ops import points_to_bev
from second.data import kitti_common as kitti
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import copy
import math
import torch


def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'gt_boxes', 'voxel_labels',
                'match_indices'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret

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

def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    class_names=['PEDESTRIAN'],
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    reduce_valid_area=False,
                    remove_unknown=False,
                    gt_rotation_noise=[-np.pi / 3, np.pi / 3],
                    gt_loc_noise_std=[1.0, 1.0, 1.0],
                    global_rotation_noise=[-np.pi / 4, np.pi / 4],
                    global_scaling_noise=[0.95, 1.05],
                    global_loc_noise_std=(0.2, 0.2, 0.2),
                    global_random_rot_range=[0.78, 2.35],
                    generate_bev=False,
                    without_reflectivity=False,
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    anchor_cache=None,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    add_rgb_to_points=False,
                    lidar_input=False,
                    unlabeled_db_sampler=None,
                    out_size_factor=2,
                    min_gt_point_dict=None,
                    bev_only=False,
                    use_group_id=False,
                    out_dtype=np.float32
                    ):
    """convert point cloud to voxels, create targets if ground truths 
    exists.
    """
    points = input_dict["points"]
    pc_range = voxel_generator.point_cloud_range
    
    pts_x, pts_y, pts_z = points[:, 0], points[:, 1], points[:, 2]
    range_flag = ( (pts_x >= pc_range[0]) & (pts_x <= pc_range[3]) 
                 & (pts_y >= pc_range[1]) & (pts_y <= pc_range[4]) 
                 & (pts_z >= pc_range[2]) & (pts_z <= pc_range[5]) )
    
    points = points[range_flag]
    
    
    if training:
        gt_boxes = input_dict["gt_boxes"]
        gt_names = input_dict["gt_names"]
        ## group_ids ?      np.arange(num_gt,dtype=np.int32) num_gt - number of objects (of all categories) in annotated lidar frame  
                
        group_ids = None
        if use_group_id and "group_ids" in input_dict:
            group_ids = input_dict["group_ids"]
    
    
    #unlabeled_training = unlabeled_db_sampler is not None
   
    if training:     
        
        
        gt_boxes_mask = np.array(
            [n in class_names for n in gt_names], dtype=np.bool_)
        #print(gt_boxes_mask.shape,gt_boxes.shape,"before")
        
        prep.noise_per_object_v3_(
            gt_boxes,
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=group_ids,
            num_try=100)
        #print(gt_boxes_mask.shape,gt_boxes.shape,"after")
        
        # should remove unrelated objects after noise per object
        gt_boxes = gt_boxes[gt_boxes_mask]
        gt_names = gt_names[gt_boxes_mask]
        
        
        if group_ids is not None:
            group_ids = group_ids[gt_boxes_mask]
        
        
        
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
        
        
        #need to check the output
        gt_boxes, points = prep.random_flip(gt_boxes, points)
        gt_boxes, points = prep.global_rotation(
            gt_boxes, points, rotation=global_rotation_noise)
        gt_boxes, points = prep.global_scaling_v2(gt_boxes, points,
                                                  *global_scaling_noise)

        # Global translation
        gt_boxes, points = prep.global_translate(gt_boxes, points, global_loc_noise_std)
        
        
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_boxes, bv_range)
        gt_boxes = gt_boxes[mask]
        gt_classes = gt_classes[mask]
        if group_ids is not None:
            group_ids = group_ids[mask]

        # limit rad to [-pi, pi]
        gt_boxes[:, 6] = box_np_ops.limit_period(
            gt_boxes[:, 6], offset=0.5, period=2 * np.pi)
        #assert -np.pi/2 <= g <= np.pi/2

    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    
    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)

    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64)
    }
    
    
    # if not lidar_input:
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    if anchor_cache is not None:
        anchors = anchor_cache["anchors"]
        anchors_bv = anchor_cache["anchors_bv"]
        matched_thresholds = anchor_cache["matched_thresholds"]
        unmatched_thresholds = anchor_cache["unmatched_thresholds"]
    else:
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
    example["anchors"] = anchors
    
    anchors_mask = None
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
        example['anchors_mask'] = anchors_mask
    
    if generate_bev:
        bev_vxsize = voxel_size.copy()
        bev_vxsize[:2] /= 2
        bev_vxsize[2] *= 2
        bev_map = points_to_bev(points, bev_vxsize, pc_range,
                                without_reflectivity)
        example["bev_map"] = bev_map
    if not training:
        return example
    if create_targets:
        targets_dict = target_assigner.assign(
            anchors,
            gt_boxes,
            anchors_mask,
            gt_classes=gt_classes,
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds)
        example.update({
            'labels': targets_dict['labels'],
            'reg_targets': targets_dict['bbox_targets'],
            'reg_weights': targets_dict['bbox_outside_weights'],
        })
    return example


def _read_and_prep_v9(info, class_names,root_path, num_point_features, argoverse_loader ,argoverse_map,prep_func):
    """read data from KITTI-format infos, then call prep function.
    """
    # velodyne_path = str(pathlib.Path(root_path) / info['velodyne_path'])
    # velodyne_path += '_reduced'
    fr_seq = info["index"]
    undr_scr = fr_seq.find('_')
    seq = int(fr_seq[:undr_scr])
    frame = int(fr_seq[undr_scr+1:])
    argoverse_data = argoverse_loader[seq]
    
    ## converting to kitti camera coordinate system
    ## all input points
    points = argoverse_data.get_lidar(frame)
    roi_pts = copy.deepcopy(points)
    objects = argoverse_data.get_label_object(frame)

    city_name = argoverse_data.city_name
    city_to_egovehicle_se3 = argoverse_data.get_pose(frame)
    am = argoverse_map
    
   
    
    if info["include_roi"] or info["dr_area"] or not info["include_road_points"]:
        roi_pts = city_to_egovehicle_se3.transform_point_cloud(roi_pts)  # put into city coords 
        
    if info["include_roi"]:
        roi_pts_flag = am.remove_non_roi_points(roi_pts, city_name) # remove non-driveable region
        roi_pts = roi_pts[roi_pts_flag]
        
    if not info["include_roi"] and info["dr_area"]:
        roi_pts_flag = am.remove_non_driveable_area_points(roi_pts, city_name) # remove non-driveable region
        roi_pts = roi_pts[roi_pts_flag]
        
        
    if not info["include_road_points"]:
        roi_pts = am.remove_ground_surface(roi_pts, city_name)  # remove ground surface

        # convert city to lidar co-ordinates
    if info["include_roi"] or info["dr_area"] or not info["include_road_points"]:
        roi_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(roi_pts) 
    
    
    roi_pts[:,2] =  roi_pts[:,2] - 1.73
    
    
    gt_location = []
    dims = []
    gt_rots = []
    gt_names = []
    
    for obj in objects:
        if obj.label_class in class_names:
            corners = obj.as_3d_bbox()
            center = np.mean(corners,axis=0)
            center[2] = corners[2,2]
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
            dims.append([obj.width,obj.length,obj.height])

    gt_location = np.array(gt_location)
    
    if info["include_roi"]:
        roi_locs = city_to_egovehicle_se3.transform_point_cloud(gt_location)  # put into city coords
        #non roi
        roi_locs_flag = am.remove_non_roi_points(roi_locs, city_name) # remove non-driveable region
    
    if not info["include_roi"] and info["dr_area"]:
        roi_locs = city_to_egovehicle_se3.transform_point_cloud(gt_location)  # put into city coords
        #non roi
        roi_locs_flag = am.remove_non_driveable_area_points(roi_locs, city_name) # remove non-driveable region
    
    
    
    dims = np.array(dims)
    gt_rots = np.array( gt_rots )
    gt_location[:,2] -= 1.73
    gt_boxes = np.concatenate([gt_location, dims, gt_rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_names = np.array(gt_names)
    
    if info["include_roi"] or info["dr_area"] :
        gt_boxes = gt_boxes[roi_locs_flag]
        gt_names = gt_names[roi_locs_flag]
    
    input_dict = {
        'points': roi_pts,
        'pointcloud_num_features': num_point_features,
        'gt_boxes': gt_boxes,
        'gt_names': gt_names,
    }

    '''
        if 'group_ids' in annos:
            input_dict['group_ids'] = annos["group_ids"]
    '''
    example = prep_func(input_dict=input_dict)
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    example["image_idx"] = fr_seq
    example["image_shape"] = np.array([400,400],dtype=np.int32)
    if info["include_roadmap"]:
        example["road_map"] = torch.load(info["road_map"] + fr_seq )
    else:
        example["road_map"] = None
    example["include_roadmap"] = info["include_roadmap"]
    #torch.save(example,"./network_input_examples/" + info)
    return example

