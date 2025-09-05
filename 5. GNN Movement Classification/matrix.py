import pandas as pd

# 读取answer.csv和submission.csv
df_true = pd.read_csv('answer.csv')
df_pred = pd.read_csv('submission.csv')

# 确保按照index对齐
df_true = df_true.sort_values('index').reset_index(drop=True)
df_pred = df_pred.sort_values('index').reset_index(drop=True)

# 计算label的accuracy
accuracy = (df_true['label'] == df_pred['label']).mean()
print(f"label的accuracy为: {accuracy:.4f}")
