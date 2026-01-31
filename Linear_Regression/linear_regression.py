"""
线性回归最小实例
使用房价预测作为示例: 根据房屋面积预测价格
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置 matplotlib 使用非交互式后端,避免 Qt 平台插件问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path):
    """加载CSV数据"""
    data = pd.read_csv(csv_path, encoding='utf-8')
    print("=" * 50)
    print("数据预览:")
    print(data.head())
    print("\n数据统计信息:")
    print(data.describe())
    return data

def prepare_data(data):
    """准备训练数据"""
    X = data[['面积(平方米)']].values  # 特征:面积
    y = data['价格(万元)'].values      # 目标:价格
    return X, y

def train_model(X_train, y_train):
    """训练线性回归模型"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def visualize_results(X, y, model, y_test, y_pred):
    """可视化结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 左图:回归线和散点图
    ax1.scatter(X, y, color='blue', alpha=0.6, label='实际数据')
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    ax1.plot(X_range, y_range, color='red', linewidth=2, label='回归线')
    ax1.set_xlabel('面积 (平方米)', fontsize=12)
    ax1.set_ylabel('价格 (万元)', fontsize=12)
    ax1.set_title('线性回归: 房屋面积 vs 价格', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图:预测值vs实际值
    ax2.scatter(y_test, y_pred, color='green', alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='完美预测线')
    ax2.set_xlabel('实际价格 (万元)', fontsize=12)
    ax2.set_ylabel('预测价格 (万元)', fontsize=12)
    ax2.set_title('预测效果: 实际值 vs 预测值', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_regression_result.png', dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存为 'linear_regression_result.png'")
    # 不再调用 plt.show(),避免显示问题

def main():
    """主函数"""
    print("线性回归示例 - 房价预测")
    print("=" * 50)

    # 1. 加载数据
    csv_path = 'linear_regression_sample.csv'
    data = load_data(csv_path)

    # 2. 准备数据
    X, y = prepare_data(data)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 4. 训练模型
    print("\n开始训练模型...")
    model = train_model(X_train, y_train)

    # 5. 获取模型参数
    slope = model.coef_[0]        # 斜率
    intercept = model.intercept_  # 截距
    print(f"\n模型参数:")
    print(f"斜率 (系数): {slope:.4f}")
    print(f"截距: {intercept:.4f}")
    print(f"\n回归方程: 价格 = {slope:.4f} × 面积 + {intercept:.4f}")

    # 6. 评估模型
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f"\n模型评估:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 7. 示例预测
    sample_areas = np.array([[75], [125], [175]])
    sample_predictions = model.predict(sample_areas)
    print(f"\n示例预测:")
    for area, price in zip(sample_areas.flatten(), sample_predictions):
        print(f"  {area} 平方米 -> 预测价格: {price:.2f} 万元")

    # 8. 可视化
    visualize_results(X, y, model, y_test, y_pred)

    print("\n" + "=" * 50)
    print("线性回归分析完成!")

if __name__ == "__main__":
    main()
