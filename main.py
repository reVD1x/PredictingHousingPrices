import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from utils import plot_feature_importance, plot_predictions

os.environ["http_proxy"] = "http://127.0.0.1:21882"
os.environ["https_proxy"] = "http://127.0.0.1:21882"

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 确保存在images文件夹
images_dir = os.path.join(os.path.dirname(__file__), 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 设置随机种子，保证结果可复现
rand_seed = 28
np.random.seed(rand_seed)

# 1. 加载数据集
print("1. 加载数据集")
house_data = fetch_california_housing()
X = house_data.data
y = house_data.target

feature_names = house_data.feature_names
feature_names_cn = ['收入中位数', '房龄', '平均房间数', '平均卧室数',
                    '街区人口', '平均入住人数', '街区纬度', '街区经度']

# 检查房价数据中5.0的值
exactly_5 = sum(y == 5.0)
greater_than_4_9 = sum(y >= 4.9)
print(f"房价等于5.0的样本数: {exactly_5}")
print(f"房价大于等于4.9的样本数: {greater_than_4_9}")
print(f"房价的最大值: {max(y)}")
print(f"房价的最小值: {min(y)}")

# 房价值的分布情况
price_values, price_counts = np.unique(y, return_counts=True)
print("\n房价值的分布情况:")
for value, count in zip(price_values[-10:], price_counts[-10:]):
    print(f"房价 = {value:.4f}: {count}个样本")

# 将数据集转换为DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y

# 创建中文列名的数据框
df_cn = df.copy()
# 重命名列为中文
for eng_name, cn_name in zip(feature_names, feature_names_cn):
    df_cn[cn_name] = df_cn[eng_name]
df_cn['房价'] = df_cn['Price']

print("数据集信息")
print(f"数据集形状: {df.shape}")
print(f"特征名称: {feature_names}")
print("数据集基本统计信息:")
print(df.describe())

# 检查缺失值
print("检查缺失值:")
print(df.isnull().sum())

# 2. 数据可视化
print("\n2. 数据可视化分析")

# 绘制房价分布图
print("绘制房价分布图")
plt.figure(figsize=(10, 6))
sns.histplot(data=df['Price'], kde=True)
plt.title('房价分布')
plt.xlabel('价格/十万美元')
plt.ylabel('频率')
plt.savefig(os.path.join(images_dir, '房价分布.png'))
plt.close()

# 数据清洗 - 去除异常值
print("\n数据清洗")
# 保存原始数据量
original_size = len(df)

# 房价为5.0的数据分析
price_5_count = sum(df['Price'] == 5.0)
price_5_percent = (price_5_count / original_size) * 100
print(f"房价为5.0的数据: {price_5_count} 个 ({price_5_percent:.2f}%)")


# 使用IQR方法检测和删除异常值
def remove_outliers(df, column):
    # 使用IQR方法检测异常值
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 检测异常值
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"列 '{column}' 中检测到 {len(outliers)} 个异常值")

    # 筛选出正常值
    result = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return result


# 对房价进行异常值处理
df = remove_outliers(df, 'Price')

# 对其他重要特征进行异常值处理
for feature in feature_names:
    df = remove_outliers(df, feature)

# 重新创建清洗后的中文列名数据框
df_cn = df.copy()
for eng_name, cn_name in zip(feature_names, feature_names_cn):
    df_cn[cn_name] = df_cn[eng_name]
df_cn['房价'] = df_cn['Price']

# 输出清洗后数据量
cleaned_size = len(df)
removed_size = original_size - cleaned_size
removed_percentage = (removed_size / original_size) * 100
print(f"原始数据量: {original_size}")
print(f"清洗后数据量: {cleaned_size}")
print(f"移除的数据量: {removed_size} ({removed_percentage:.2f}%)")
print(f"清洗后房价为5.0的数据点: {sum(df['Price'] == 5.0)} 个")

# 清洗后的房价分布图
print("绘制清洗后房价分布图")
plt.figure(figsize=(10, 6))
sns.histplot(data=df['Price'], kde=True)
plt.title('清洗后房价分布')
plt.xlabel('房价/十万美元')
plt.ylabel('频率')
plt.savefig(os.path.join(images_dir, '清洗后房价分布.png'))
plt.close()

# 绘制不同特征与房价的散点图
print("绘制不同特征与房价的散点图")
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()

for i, feature in enumerate(feature_names_cn):
    sns.scatterplot(x=feature, y='房价', data=df_cn, alpha=0.5, ax=axes[i])
    axes[i].set_title(f'{feature}-房价')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '特征与房价关系.png'))
plt.close()

# 3. 训练数据预处理
print("\n3. 训练数据预处理")
X = df.drop('Price', axis=1).values
y = df['Price'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 模型训练与评估
print("\n4. 模型训练与评估")

# 模型结果
models = {}
model_names = []
model_names_cn = []
mse_scores = []
rmse_scores = []
r2_scores = []


def evaluate_model(model, name, name_cn, X_train, X_test, y_train, y_test, scaled=False):
    """评估模型性能"""
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name_cn}模型评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # 存储结果
    models[name] = model
    model_names.append(name)
    model_names_cn.append(name_cn)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    # 保存预测-实际值散点图
    data_to_plot = {
        'X_test': X_test if not scaled else X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'name': name,
        'name_cn': name_cn
    }
    return data_to_plot


# 4.1 简单线性回归
print("\n4.1 简单线性回归")
lr_model = LinearRegression()
lr_data = evaluate_model(lr_model, "Linear_Regression", "简单线性回归",
                         X_train_scaled, X_test_scaled, y_train, y_test, scaled=True)

# 4.2 多项式回归
print("\n4.2 多项式回归")
degree = 2  # 多项式的阶数
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_data = evaluate_model(poly_model, f"Polynomial_Regression_degree{degree}",
                           f"多项式回归({degree}阶)",
                           X_train_poly, X_test_poly, y_train, y_test, scaled=True)

# 4.3 随机森林
print("\n4.3 随机森林")
rf_model = RandomForestRegressor(n_estimators=100, random_state=rand_seed)
rf_data = evaluate_model(rf_model, "Random_Forest", "随机森林",
                         X_train, X_test, y_train, y_test)

# 查看特征重要性
print("绘制随机森林模型特征重要性")
feature_importance = rf_model.feature_importances_
plot_feature_importance(feature_importance, feature_names_cn,
                        "随机森林模型特征重要性",
                        os.path.join(images_dir, "随机森林特征重要性.png"))

# 5. 模型比较可视化
print("\n5. 模型性能比较")

# 绘制不同模型比较
print("绘制不同模型比较")
plt.figure(figsize=(10, 15))

# 绘制不同模型的MSE比较
plt.subplot(3, 1, 1)
sns.barplot(x=model_names_cn, y=mse_scores)
plt.title('不同模型的MSE比较 (越低越好)')
plt.xticks(rotation=45)

# 绘制不同模型的RMSE比较
plt.subplot(3, 1, 2)
sns.barplot(x=model_names_cn, y=rmse_scores)
plt.title('不同模型的RMSE比较 (越低越好)')
plt.xticks(rotation=45)

# 绘制不同模型的R^2比较
plt.subplot(3, 1, 3)
sns.barplot(x=model_names_cn, y=r2_scores)
plt.title('不同模型的R^2比较 (越高越好)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '模型比较.png'))
plt.close()

# 为各个模型生成预测图
print("绘制预测-实际值图")
plot_predictions(lr_data, "简单线性回归预测-实际值")
plot_predictions(poly_data, "多项式回归预测-实际值")
plot_predictions(rf_data, "随机森林预测-实际值")

# 保存最佳模型
best_model_idx = np.argmax(r2_scores)
best_model_name = model_names[best_model_idx]
best_model_name_cn = model_names_cn[best_model_idx]
print(f"\n最佳模型为: {best_model_name_cn}，R^2评分: {max(r2_scores):.4f}")
joblib.dump(models[best_model_name], os.path.join(images_dir, '最佳模型.pkl'))

print("项目完成")
