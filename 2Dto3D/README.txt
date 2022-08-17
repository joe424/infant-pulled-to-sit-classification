* 檔案'config.ini'設定下面流程的資料夾的路徑
    * 流程如下
        * 放在'INPUT_FOLDER_PATH'資料夾中的json檔案會由'2Dto3D.py'進行2D轉3D
        * 使用'MODEL_FOLDER_PATH'資料夾中的模型進行2D轉3D
        * 輸出的3D骨架會儲存至'OUTPUT3D_FOLDER_PATH'資料夾
        * 輸出的投射回2D的骨架會儲存至'OUTPUT2D_FOLDER_PATH'資料夾
* 2Dto3D.py
    將2D骨架轉為3D骨架和投影回2D骨架
* input/
    放2D骨架(.json)讓2Dto3D.py將其轉為3D骨架和投影回2D骨架
* output2d/
    放投影回2D的骨架
    將裡面檔案放到../classfication/samples之中以分類其level
* output3d/
    放轉換成3D的骨架
* template.npy
    3D骨架要對齊的模板
* model/
    裡面放2D轉3D的模型
    
* npy形狀
    * 3D => (96, 13, 3) => (frame, joint, xyz)
    * 2D => (96, 13, 2) => (frame, joint, xy)
* 關節點號碼
    * 0    1     2     3    4     5     6    7         8      9      10        11     12
    * rhip rknee rfoot lhip lknee lfoot nose lshoulder lelbow lwrist rshoulder relbow rwrist