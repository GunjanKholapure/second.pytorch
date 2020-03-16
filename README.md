For point_pillars installation refer to https://github.com/nutonomy/second.pytorch.

The code has been modified to run on Argoverse 3D dataset. Changes are done mainly in dataloading part in point_pillars/second/data/preprocess.py

Command to train final model (in point_pillars/second/) - 

python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_20_argo_upper.proto --model_dir=./models --device=0 --include_roi=True --dr_area=False --include_road_points=False

Pre-trained model can be downloaded from -
https://drive.google.com/file/d/1ebtmoQSJdfIKVVJ93GSCdeGy-ZGqGXyN/view?usp=sharing


For inference on test set (in point_pillars/second)- 

python pp_inference.py --config_path=./configs/pointpillars/car/xyres_20_argo_upper.proto --model_dir=./models   --device=0 --model_path="path_to_model/voxelnet-xxx.tckpt" --save_path="path_to_save/xxx.pkl" --include_roi=1 --include_road_points=0 --dr_area=0

Command to get Results in AB3DMOT format( in point_pilllars/second) -

python get_tracking_result.py --model_path=path_to_model --sv_dir=path_to_AB3DMOT/data/argo/car_3d_det_val/ --set=val (or test)

