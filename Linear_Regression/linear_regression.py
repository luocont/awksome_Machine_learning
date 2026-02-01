"""
线性回归最小实例
使用房价预测作为示例: 根据房屋面积预测价格
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

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

def create_animation(X, y):
    """创建线性回归动画 - 逐步添加数据点"""
    print("\n" + "=" * 50)
    print("开始生成线性回归动画...")
    print("=" * 50)

    # 创建保存帧的目录
    frames_dir = 'animation_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 打乱数据顺序,使动画更有趣
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    print(f"总共有 {len(X)} 个数据点")

    # 为每个数据点数量生成一帧(至少需要2个点才能拟合)
    for n_points in range(2, len(X_shuffled) + 1):
        fig, ax = plt.subplots(figsize=(12, 8))

        # 当前数据点
        x_current = X_shuffled[:n_points].flatten()
        y_current = y_shuffled[:n_points]

        # 绘制所有可能的点(灰色显示未添加的)
        ax.scatter(X_shuffled[n_points:].flatten(), y_shuffled[n_points:],
                  c='lightgray', s=80, alpha=0.3, edgecolors='gray', label='待添加点')

        # 绘制当前已有的点
        ax.scatter(x_current[:-1], y_current[:-1], c='blue', s=100,
                  alpha=0.7, edgecolors='black', linewidth=1, label='已有数据点', zorder=4)

        # 最新添加的点高亮显示
        ax.scatter([x_current[-1]], [y_current[-1]], s=300, c='red',
                  alpha=0.8, edgecolors='yellow', linewidth=3, zorder=5, label='新增点')

        # 训练线性回归
        model = LinearRegression()
        model.fit(x_current.reshape(-1, 1), y_current)

        # 绘制回归线
        x_min, x_max = X.min() - 5, X.max() + 5
        x_line = np.array([[x_min], [x_max]])
        y_line = model.predict(x_line)
        ax.plot(x_line, y_line, 'b-', linewidth=3, label='回归线', zorder=3)

        # 计算统计信息
        y_pred = model.predict(x_current.reshape(-1, 1))
        mse = np.mean((y_current - y_pred) ** 2)
        ss_res = np.sum((y_current - y_pred) ** 2)
        ss_tot = np.sum((y_current - np.mean(y_current)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 添加信息框
        info_text = f'数据点数: {n_points}/{len(X)}\n'
        info_text += f'回归方程: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}\n'
        info_text += f'MSE: {mse:.4f}\n'
        info_text += f'R2: {r2:.4f}'

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        # 设置图表属性
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y.min() - 20, y.max() + 20)
        ax.set_xlabel('面积 (平方米)', fontsize=14)
        ax.set_ylabel('价格 (万元)', fontsize=14)
        ax.set_title(f'线性回归动态演示 - 第 {n_points} 个数据点', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存这一帧
        frame_filename = os.path.join(frames_dir, f'frame_{n_points:03d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if n_points % 5 == 0 or n_points == len(X):
            print(f"  已生成 {n_points}/{len(X)} 帧")

    print(f"\n所有帧已保存到: {frames_dir}/")

    # 尝试生成GIF
    try:
        from PIL import Image
        print("\n正在生成GIF动画...")

        frames = []
        for n_points in range(2, len(X_shuffled) + 1):
            frame_filename = os.path.join(frames_dir, f'frame_{n_points:03d}.png')
            img = Image.open(frame_filename)
            frames.append(img)

        # 保存为GIF
        gif_path = 'linear_regression_animation.gif'
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=600,  # 每帧600毫秒
                       loop=0)  # 无限循环

        print(f"✅ GIF动画已保存为: {gif_path}")

    except ImportError:
        print("⚠️  PIL未安装,无法生成GIF")
        print("   安装方法: pip install Pillow")

    print("\n📊 动画说明:")
    print("- 蓝色点: 已添加的数据点")
    print("- 红色光圈: 最新添加的点")
    print("- 灰色点: 待添加的数据点")
    print("- 蓝色线: 当前学习到的回归线")
    print("- 观察回归线如何随着数据增加而变化")

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

    # 9. 生成动画
    create_animation(X, y)

    print("\n" + "=" * 50)
    print("线性回归分析完成!")

if __name__ == "__main__":
    main()
