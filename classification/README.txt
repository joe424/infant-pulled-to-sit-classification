* KMU_hospital_infant_pulled_to_sit_data/
    * 詳細請見裡面的README.txt
* graph/
    * HA-GCN的程式碼
* model/
    * HA-GCN的程式碼
* samples/
    * 放投影回2D骨架的地方
* weights/
    * pulled-to-sit level分類的網路
        * model.pt
            * 由 KMU_hospital_infant_pulled_to_sit_data/pulled-to-sit_2022/ 訓練而來
        * model1.pt
            * 由 KMU_hospital_infant_pulled_to_sit_data/pulled-to-sit/ 訓練而來
* inference.py
    * 將投影回2D骨架做pulled-to-sit的分類
* train.py
    * 訓練HA-GCN，pulled-to-sit分類網路