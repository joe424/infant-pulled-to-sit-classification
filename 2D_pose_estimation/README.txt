* frames/中會儲存影片轉成的圖片
* out/
    * extracted_skeletons/
        * pkl檔
        * 儲存估測出的骨架資料
        * 2D轉3D時會使用該檔案
        * 我沒有使用
    * jsons/
        * json檔
        * 基本上裡面的東西和pkl檔相同
        * 比pkl檔多一個資訊:估測出的關節點的confidence
    * visualize/
        * 估測出的骨架弄成圖片(把./lighttrack/demo_video_mobile_hrnet.py中的flag_visualize設為True，會比較耗時)
    * videos/
        * 把上面資料夾的檔案串成影片(把./lighttrack/demo_video_mobile_hrnet.py中的flag_visualize設為True，會比較耗時)
* inference.py將videos/pull_to_sit/中所有影片估測出2D骨架
* ./lighttrack/video_to_frames.py
    * 將video變為一張一張的frame
* ./lighttrack/demo_video_mobile_hrnet.py
    * detect嬰兒
    * estimate嬰兒2D骨架，並儲存成pkl、json、圖片、影片
* lighttrack/detector/detector_yolov3.py
    * infant detector
    * 使用https://github.com/eriklindernoren/PyTorch-YOLOv3做為訓練的code
    * 使用lighttrack/weights/yolov3_ckpt161_subdiv_4_lr_0_0001_dev_0.pth做為weight
* lighttrack/dcpose/tools/inference.py
    * 2D pose estimation model
    * 使用https://github.com/Pose-Group/DCPose中的HRNet做為訓練的code
        * config可見./lighttrack/dcpose/configs/posetimation/DcPose/posetrack18/model_HRnet.yaml
    * 使用lighttrack/weights/epoch_37_state_optimize_HRNet.pth做為weight
* 缺什麼package裝什麼
* 在安裝package中比較特別的是torchlight，不能透過pip或conda，要透過cd ./lighttrack/graph/torchlight，再python setup.py install