import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def plot_feature_importance(feature_importance, feature_names, title, filename):
    """
    绘制特征重要性图

    参数:
    - feature_importance: 特征重要性数组
    - feature_names: 特征名称列表
    - title: 图表标题
    - filename: 保存的文件名
    """
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # 按重要性降序排序
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # 绘制条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_predictions(data, title):
    """
    绘制预测值与实际值的散点图

    参数:
    - data: 包含X_test, y_test, y_pred和name的字典
    - title: 图表标题
    """
    plt.figure(figsize=(10, 6))

    # 绘制预测值与实际值的散点图
    plt.scatter(data['y_test'], data['y_pred'], alpha=0.5)

    # 添加对角线
    min_val = min(data['y_test'].min(), data['y_pred'].min())
    max_val = max(data['y_test'].max(), data['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(title)
    plt.tight_layout()

    # 确保images文件夹存在
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # 保存图像
    filename = os.path.join(images_dir, f"{data['name']}预测结果.png")
    plt.savefig(filename)
    plt.close()
