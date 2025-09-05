import numpy as np
import pandas as pd

# 读取test数据
data = np.load('./train_data/FS_data_for_HDGCN_jump.npz')
# 直接获取test集的真实标签
y_test = data['y_test']  # 假设y_test为one-hot编码
# 转为类别索引
labels = np.argmax(y_test, axis=1)

# 分别保存index和label到answer.csv
df = pd.DataFrame({'index': np.arange(len(labels)), 'label': labels})
df.to_csv('answer.csv', index=False)
print("index和真实标签已保存到 answer.csv")
