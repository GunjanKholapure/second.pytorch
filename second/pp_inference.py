pth = '/users/gpu/gunji/.local/lib/python3.6/site-packages'
import sys
if pth in sys.path:
    sys.path.remove(pth)

print("pp_inference",sys.path)
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import pathlib
import shutil
from second.data.preprocess import merge_second_batch, prep_pointcloud
from pathlib import Path

import numpy as np
import torch
import os

import torchplus
from second.core import box_np_ops
from second.core.inference import InferenceContext
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder, second_builder
from second.pytorch.models.voxelnet import VoxelNet
from second.pytorch.train import predict_kitti_to_anno, example_convert_to_torch
import fire
import copy
import abc
import contextlib

import numpy as np
from google.protobuf import text_format

from second.data.preprocess import merge_second_batch, prep_pointcloud
from second.protos import pipeline_pb2
from numba.errors import NumbaDeprecationWarning, NumbaWarning, NumbaPerformanceWarning
import warnings
import pickle
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


import argparse
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--device', type=int, default=1, help='specify the gpu device for training')
parser.add_argument('--model_path', type=str, help='specify path to load model from.')
parser.add_argument('--save_path', type=str, help='specify path to save evaluation results.')
parser.add_argument('--config_path', type=str, help='config path')
parser.add_argument('--model_dir', type=str, help='model dir')
parser.add_argument('--include_roi',type=int,default=1)
parser.add_argument('--include_road_points',type=int,default=0)
parser.add_argument('--dr_area',type=int,default=0)
parser.add_argument('--set', default='val',type=str, help='val/test/train')



args = parser.parse_args()
print(args.device)
torch.cuda.set_device(args.device)



def test(config_path=args.config_path,
          model_dir = args.model_dir,
          result_path=None,
          create_folder=False,
          pickle_result=True,
         include_roadmap=False,
         device=1):
    """train a VoxelNet model specified by a config file.
    """
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    batch_size = 1
    class_names = list(input_cfg.class_names)
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    grid_size = voxel_generator.grid_size
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    net = second_builder.build(model_cfg, voxel_generator, target_assigner,include_roadmap)
    net.cuda().eval()
    
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    
    #torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    torchplus.train.restore(args.model_path, net)
    #torchplus.train.restore("./ped_models_56/voxelnet-275130.tckpt",net)
    out_size_factor = model_cfg.rpn.layer_strides[0] / model_cfg.rpn.upsample_strides[0]
    print(out_size_factor)
    #out_size_factor *= model_cfg.middle_feature_extractor.downsample_factor
    out_size_factor = int(out_size_factor)
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    print(feature_map_size)
    ret = target_assigner.generate_anchors(feature_map_size)
    #anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
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
    #"anchors_dict": anchors_dict,
    }
    
    
    am = ArgoverseMap()
    dt_annos = []
    
    root_dir =  os.path.join('./../../argodataset/argoverse-tracking/',args.set)
    argoverse_loader = ArgoverseTrackingLoader(root_dir)

    prog_cnt = 0
    for seq in range(len(argoverse_loader)):
        argoverse_data = argoverse_loader[seq]
        nlf = argoverse_data.num_lidar_frame
        for frame in range(nlf):
            prog_cnt += 1
            if prog_cnt%50 ==0:
                print(prog_cnt)
            points = argoverse_data.get_lidar(frame)
            roi_pts = copy.deepcopy(points)        
            city_name = argoverse_data.city_name
            city_to_egovehicle_se3 = argoverse_data.get_pose(frame)
    
            
            '''
            roi_pts = city_to_egovehicle_se3.transform_point_cloud(roi_pts)  # put into city coords
            #non roi
            roi_pts_flag = am.remove_non_roi_points(roi_pts, city_name) # remove non-driveable region
            roi_pts = roi_pts[roi_pts_flag]
            roi_pts = am.remove_ground_surface(roi_pts, city_name)  # remove ground surface
    
            # convert city to lidar co-ordinates

            roi_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(roi_pts) 
            '''
            if args.include_roi or args.dr_area or not args.include_road_points:
                roi_pts = city_to_egovehicle_se3.transform_point_cloud(roi_pts)  # put into city coords 
        
            if args.include_roi:
                roi_pts_flag = am.remove_non_roi_points(roi_pts, city_name) # remove non-driveable region
                roi_pts = roi_pts[roi_pts_flag]
        
            if not args.include_roi and args.dr_area:
                roi_pts_flag = am.remove_non_driveable_area_points(roi_pts, city_name) # remove non-driveable region
                roi_pts = roi_pts[roi_pts_flag]
        
        
            if not args.include_road_points:
                roi_pts = am.remove_ground_surface(roi_pts, city_name)  # remove ground surface

            # convert city to lidar co-ordinates
            if args.include_roi or args.dr_area or not args.include_road_points:
                roi_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(roi_pts) 
            
            
            
            
            roi_pts[:,2] =  roi_pts[:,2] - 1.73

            pts_x, pts_y, pts_z = roi_pts[:,0], roi_pts[:,1], roi_pts[:,2]
    
     
            input_dict = {
                'points': roi_pts,
                'pointcloud_num_features': 3,
            }
            
            out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
            
            
            example = prep_pointcloud(
            input_dict=input_dict,
            root_path=None,
            voxel_generator=voxel_generator,
            target_assigner=target_assigner,
            max_voxels=input_cfg.max_number_of_voxels,
            class_names=list(input_cfg.class_names),
            training=False,
            create_targets=False,
            shuffle_points=input_cfg.shuffle_points,
            generate_bev=False,
            without_reflectivity=model_cfg.without_reflectivity,
            num_point_features=model_cfg.num_point_features,
            anchor_area_threshold=input_cfg.anchor_area_threshold,
            anchor_cache=anchor_cache,
            out_size_factor=out_size_factor,
            out_dtype=np.float32)
            
            
            
            if "anchors_mask" in example:
                example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
            example["image_idx"] = str(seq) + "_" + str(frame)
            example["image_shape"] = np.array([400,400],dtype=np.int32)
            example["road_map"] = None
            example["include_roadmap"] = False
            example["points"] = roi_pts
            #torch.save(example,"./network_input_examples/" + info)
            example = merge_second_batch([example])
            
            example_torch = example_convert_to_torch(example,device=args.device)
            try:
                result_annos = predict_kitti_to_anno(
                net, example_torch, input_cfg.class_names,
                model_cfg.post_center_limit_range, model_cfg.lidar_input)
            except:
                print(seq,frame)
                continue
            dt_annos += result_annos
            
    if pickle_result:
        sdi = args.save_path.rfind('/')
        save_dir = args.save_path[:sdi]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        with open( args.save_path, 'wb') as f:
            pickle.dump(dt_annos, f)
            
  
if __name__ == '__main__':
    test()
    


    