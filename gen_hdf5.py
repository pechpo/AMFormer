import pandas as pd
import h5py
import numpy as np

# 读取CSV文件到Pandas DataFrame
csv_file = "final_feature_2.csv"
df = pd.read_csv(csv_file)

# 转换为NumPy数组
data = df.values

# 获取标签列（最后一列是标签）
labels = data[:, -1].astype(int)
data = data[:, :-1]
#print(data.shape)

def map_labels(labels):
    mapped_labels = np.zeros_like(labels)
    mapped_labels[(labels >= 0) & (labels <= 59)] = 0
    mapped_labels[(labels >= 60) & (labels <= 79)] = 1
    mapped_labels[(labels >= 80) & (labels <= 100)] = 2
    return mapped_labels

labels = map_labels(labels)

# 创建HDF5文件并写入数据
with h5py.File("lexue_data.hdf5", "w") as f:
    f.create_dataset("my_dataset", data=data)
    f.create_dataset("label", data=labels)

print("HDF5文件已成功创建。")
