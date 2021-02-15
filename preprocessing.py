import pandas as pd

def preprocessingDriftData():
    file_path = "dataset/stream_drift.csv"
    drift_data_frame = pd.read_csv(file_path, sep=',', header=0).iloc[:,1:]
    drift_data_frame['label'] = 1
    print(drift_data_frame)
    return drift_data_frame


def preprocessingNormalData():
    file_path = "dataset/stream_normal.csv"
    normal_data_frame = pd.read_csv(file_path, sep=',', header=0).iloc[:,1:]
    normal_data_frame['label'] = 0

    print(normal_data_frame)
    return normal_data_frame


# if __name__ == "__main__":
#     feature_length = 200
#     all_drift_data_frame = preprocessingDriftData()
#     all_data_frame = all_drift_data_frame
#     print(all_data_frame.shape)
#     all_normal_data_frame = preprocessingNormalData()
#     print(all_normal_data_frame.shape)
#     all_data_frame = pd.concat([all_data_frame, all_normal_data_frame], ignore_index=True)
#     print(all_data_frame.shape)
#
#     columnList = []
#     for i in range(feature_length):
#         feature_name = "feature_" + str(i+1)
#         columnList.append(feature_name)
#     columnList.append('label')
#     all_data_frame.columns = columnList
#
#     print(all_data_frame)
#     print(all_data_frame.shape)


def LoadDriftData(Data_Vector_Length, DATA_FILE):
    # feature_length = Data_Vector_Length
    # sample_num = 1250
    #
    feature_length = 200
    all_drift_data_frame = preprocessingDriftData()
    all_data_frame = all_drift_data_frame
    all_normal_data_frame = preprocessingNormalData()
    all_data_frame = pd.concat([all_data_frame, all_normal_data_frame], ignore_index=True)

    columnList = []
    for i in range(feature_length):
        feature_name = "feature_" + str(i)
        columnList.append(feature_name)
    columnList.append('label')
    all_data_frame.columns = columnList
    return all_data_frame