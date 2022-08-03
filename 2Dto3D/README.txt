* Python 3.7.13
* Pytorch 1.7.1+cu110
* 檔案'config.ini'設定下面流程的資料夾的路徑
* 流程如下
    * 放在'INPUT_FOLDER_PATH'資料夾中的pkl檔案會由'2Dto3D.py'進行2D轉3D
    * 使用'MODEL_FOLDER_PATH'資料夾中的模型進行2D轉3D
    * 輸出的3D骨架會儲存至'OUTPUT3D_FOLDER_PATH'資料夾
    * 輸出的投射回2D的骨架會儲存至'OUTPUT2D_FOLDER_PATH'資料夾
* npy形狀
    * 3D => (96, 13, 3) => (frame, joint, xyz)
    * 2D => (96, 13, 2) => (frame, joint, xy)
* 關節點號碼
    * 0    1     2     3    4     5     6    7         8      9      10        11     12
    * rhip rknee rfoot lhip lknee lfoot nose lshoulder lelbow lwrist rshoulder relbow rwrist