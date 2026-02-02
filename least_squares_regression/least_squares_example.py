"""
最小二乘回归算法最小实例
使用房价预测数据作为示例: 根据房屋面积预测房价
展示最小二乘法的几何意义和优化过程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path):
    """加载CSV数据"""
    data = pd.read_csv(csv_path, encoding='utf-8')
    return data

def prepare_data(data):
    """准备数据"""
    # 特征列
    feature_columns = ['面积', '卧室数', '房龄']

    X = data[feature_columns].values
    y = data['房价'].values

    return X, y, feature_columns

def normal_equation(X, y):
    """正规方程求解最小二乘回归
    θ = (X^T * X)^(-1) * X^T * y
    """
    # 添加偏置项（截距）
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # 正规方程求解
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta

def gradient_descent(X, y, learning_rate=0.0001, n_iterations=1000):
    """梯度下降求解最小二乘回归"""
    m, n = X.shape

    # 添加偏置项
    X_b = np.c_[np.ones((m, 1)), X]

    # 初始化参数
    theta = np.random.randn(n + 1)

    # 记录损失历史
    cost_history = []

    for iteration in range(n_iterations):
        # 计算预测值
        predictions = X_b.dot(theta)

        # 计算误差
        errors = predictions - y

        # 计算梯度
        gradients = (2/m) * X_b.T.dot(errors)

        # 更新参数
        theta = theta - learning_rate * gradients

        # 计算损失（MSE）
        cost = (1/m) * np.sum(errors ** 2)
        cost_history.append(cost)

    return theta, cost_history

def predict(X, theta):
    """预测"""
    # 添加偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

def visualize_regression_line(X, y, theta, feature_columns, save_path='ls_regression_line.png'):
    """可视化回归线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 对每个特征绘制回归线
    for idx, ax in enumerate(axes):
        feature_idx = idx
        feature_name = feature_columns[feature_idx]

        # 提取当前特征
        x_feature = X[:, feature_idx]

        # 绘制散点图
        ax.scatter(x_feature, y, alpha=0.6, s=50, c='#3498db',
                  edgecolors='black', linewidth=0.5, label='实际数据')

        # 绘制回归线
        x_range = np.linspace(x_feature.min(), x_feature.max(), 100)
        # 使用其他特征的均值来构建预测
        X_pred = np.zeros((len(x_range), len(feature_columns)))
        for i in range(len(feature_columns)):
            if i == feature_idx:
                X_pred[:, i] = x_range
            else:
                X_pred[:, i] = X[:, i].mean()

        y_pred = predict(X_pred, theta)
        ax.plot(x_range, y_pred, 'r-', linewidth=3, label='回归线')

        ax.set_xlabel(f'{feature_name}', fontsize=12)
        ax.set_ylabel('房价 (万元)', fontsize=12)
        ax.set_title(f'房价 vs {feature_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"回归线图已保存为 '{save_path}'")
    plt.close()

def visualize_residuals(y_true, y_pred, save_path='ls_residuals.png'):
    """可视化残差分析"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 残差散点图
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, c='#e74c3c',
               edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('预测值', fontsize=12)
    ax1.set_ylabel('残差', fontsize=12)
    ax1.set_title('残差 vs 预测值', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # 2. 残差直方图
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('残差', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('残差分布', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # 3. Q-Q图（正态性检验）
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q图 (正态性检验)', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)

    # 4. 残差顺序图
    ax4 = axes[1, 1]
    ax4.plot(range(len(residuals)), residuals, 'o-', alpha=0.6, markersize=4)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('样本索引', fontsize=12)
    ax4.set_ylabel('残差', fontsize=12)
    ax4.set_title('残差顺序图', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"残差分析图已保存为 '{save_path}'")
    plt.close()

def visualize_loss_surface(X, y, feature_columns, save_path='ls_loss_surface.png'):
    """可视化损失函数曲面"""
    # 只使用前两个特征进行可视化
    X_2d = X[:, :2]
    feature_x = feature_columns[0]
    feature_y = feature_columns[1]

    # 训练简化模型（只用两个特征）
    X_b = np.c_[np.ones((X_2d.shape[0], 1)), X_2d]
    theta_opt = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # 创建参数网格
    theta0_range = np.linspace(theta_opt[0] - 50, theta_opt[0] + 50, 50)
    theta1_range = np.linspace(theta_opt[1] - 1, theta_opt[1] + 1, 50)
    theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)

    # 计算损失
    loss_grid = np.zeros_like(theta0_grid)
    for i in range(theta0_grid.shape[0]):
        for j in range(theta0_grid.shape[1]):
            theta = np.array([theta0_grid[i, j], theta1_grid[i, j], theta_opt[2]])
            predictions = X_b.dot(theta)
            loss = np.mean((predictions - y) ** 2)
            loss_grid[i, j] = loss

    fig = plt.figure(figsize=(16, 6))

    # 3D曲面图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(theta0_grid, theta1_grid, loss_grid,
                           cmap='viridis', alpha=0.8)
    ax1.scatter([theta_opt[0]], [theta_opt[1]], [np.mean((X_b.dot(theta_opt) - y)**2)],
               color='red', s=200, marker='*', label='最优解')
    ax1.set_xlabel('theta_0 (截距)', fontsize=11)
    ax1.set_ylabel(f'theta_1 ({feature_x}系数)', fontsize=11)
    ax1.set_zlabel('损失 (MSE)', fontsize=11)
    ax1.set_title('损失函数曲面', fontsize=13, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    ax1.legend()

    # 等高线图
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(theta0_grid, theta1_grid, loss_grid, levels=20, cmap='viridis')
    ax2.scatter([theta_opt[0]], [theta_opt[1]], color='red', s=200,
               marker='*', label='最优解', zorder=5)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('theta_0 (截距)', fontsize=11)
    ax2.set_ylabel(f'theta_1 ({feature_x}系数)', fontsize=11)
    ax2.set_title('损失函数等高线', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"损失曲面图已保存为 '{save_path}'")
    plt.close()

def visualize_least_squares_geometry(X, y, theta, feature_columns, save_path='ls_geometry.png'):
    """可视化最小二乘法的几何意义"""
    # 只使用前两个特征进行3D可视化
    X_2d = X[:, :2]
    feature_x = feature_columns[0]
    feature_y = feature_columns[1]

    # 添加偏置项
    X_b = np.c_[np.ones((X_2d.shape[0], 1)), X_2d]

    # 计算预测值
    y_pred = X_b.dot(theta[:3])

    fig = plt.figure(figsize=(18, 6))

    # 子图1: 3D散点图和回归平面
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    # 绘制数据点
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], y, c='#3498db', marker='o',
               s=50, alpha=0.6, label='实际值', edgecolors='black')

    # 绘制回归平面
    x1_range = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), 20)
    x2_range = np.linspace(X_2d[:, 1].min(), X_2d[:, 1].max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # 构建平面网格
    X_grid = np.column_stack([np.ones(x1_grid.size).flatten(),
                              x1_grid.flatten(),
                              x2_grid.flatten()])
    y_grid = X_grid.dot(theta[:3]).reshape(x1_grid.shape)

    ax1.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3, color='#e74c3c')

    # 绘制残差线
    for i in range(0, len(X_2d), 5):  # 每隔5个点画一条残差线
        ax1.plot([X_2d[i, 0], X_2d[i, 0]],
                [X_2d[i, 1], X_2d[i, 1]],
                [y[i], y_pred[i]], 'g-', linewidth=1)

    ax1.set_xlabel(f'{feature_x}', fontsize=11)
    ax1.set_ylabel(f'{feature_y}', fontsize=11)
    ax1.set_zlabel('房价', fontsize=11)
    ax1.set_title('3D回归平面', fontsize=13, fontweight='bold')
    ax1.legend()

    # 子图2: 投影到X空间的几何解释
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # 在这个视图中，我们展示y向量在X列空间上的投影
    # 简化：只展示前两个样本的向量
    n_samples_show = min(10, len(X_2d))

    # 绘制y向量
    ax2.quiver(0, 0, 0, 0, 0, y[0], color='blue', linewidth=3, label='y向量')
    ax2.quiver(0, 0, 0, 0, 0, y_pred[0], color='red', linewidth=3, label='投影向量 Xθ')
    ax2.quiver(0, 0, y_pred[0], 0, 0, y[0] - y_pred[0],
              color='green', linewidth=3, label='残差向量 (垂直)')

    ax2.set_xlabel('X₁', fontsize=11)
    ax2.set_ylabel('X₂', fontsize=11)
    ax2.set_zlabel('y', fontsize=11)
    ax2.set_title('最小二乘几何意义', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])

    # 子图3: 正规方程示意
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')

    explanation = f"""
    最小二乘法的几何意义

    目标: 最小化 ||y - Xθ||^2

    正规方程:
    X^T X θ = X^T y

    解:
    θ = (X^T X)^(-1) X^T y

    几何解释:
    - 蓝色向量: y (观测值)
    - 红色向量: Xθ (预测值)
    - 绿色向量: 残差

    关键性质:
    残差向量 ⊥ X的列空间

    即: X^T (y - Xθ) = 0

    这保证了预测值是y在X列空间
    上的正交投影，使残差最小。
    """

    ax3.text(0.1, 0.5, explanation, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"几何意义图已保存为 '{save_path}'")
    plt.close()

def visualize_model_comparison(X_train, X_test, y_train, y_test, theta,
                              feature_columns, save_path='ls_model_comparison.png'):
    """可视化模型性能对比"""
    # 预测
    y_train_pred = predict(X_train, theta)
    y_test_pred = predict(X_test, theta)

    # 计算指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 训练集预测 vs 实际
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_train_pred, alpha=0.6, s=50, c='#3498db',
               edgecolors='black', linewidth=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
            'r--', linewidth=2, label='完美预测线')
    ax1.set_xlabel('实际房价', fontsize=12)
    ax1.set_ylabel('预测房价', fontsize=12)
    ax1.set_title(f'训练集: R^2={train_r2:.4f}, MSE={train_mse:.2f}',
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. 测试集预测 vs 实际
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50, c='#e74c3c',
               edgecolors='black', linewidth=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='完美预测线')
    ax2.set_xlabel('实际房价', fontsize=12)
    ax2.set_ylabel('预测房价', fontsize=12)
    ax2.set_title(f'测试集: R^2={test_r2:.4f}, MSE={test_mse:.2f}',
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 指标对比柱状图
    ax3 = axes[1, 0]
    metrics = ['MSE', 'R^2', 'MAE']
    train_values = [train_mse, train_r2, train_mae]
    test_values = [test_mse, test_r2, test_mae]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, train_values, width, label='训练集',
                   color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, test_values, width, label='测试集',
                   color='#e74c3c', alpha=0.8)

    ax3.set_xlabel('评估指标', fontsize=12)
    ax3.set_ylabel('值', fontsize=12)
    ax3.set_title('模型性能对比', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. 系数解释
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 显示回归系数
    coef_text = f"""
    回归系数解释

    截距 (theta_0): {theta[0]:.2f}

    特征系数:
    """
    for i, feat in enumerate(feature_columns):
        coef_text += f"  {feat} (theta_{i+1}): {theta[i+1]:.4f}\n"

    coef_text += f"""
    模型解释:

    - 房价 = {theta[0]:.2f}
    """
    for i, feat in enumerate(feature_columns):
        sign = "+" if theta[i+1] >= 0 else ""
        coef_text += f" {sign} {theta[i+1]:.4f}×{feat}"
    coef_text += "\n"

    coef_text += f"""
    - R^2 = {test_r2:.4f}
      模型解释了 {test_r2*100:.1f}% 的房价变异

    - 每增加1平方米，房价变化 {theta[1]:.2f} 万元

    - 每增加1个卧室，房价变化 {theta[2]:.2f} 万元

    - 每增加1年房龄，房价变化 {theta[3]:.2f} 万元
    """

    ax4.text(0.1, 0.5, coef_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"模型对比图已保存为 '{save_path}'")
    plt.close()

def create_animation(X, y, feature_columns, max_iterations=50):
    """创建最小二乘回归动画 - 展示梯度下降优化过程"""
    print("\n" + "=" * 70)
    print("开始生成最小二乘回归动画...")
    print("=" * 70)

    # 创建保存帧的目录
    frames_dir = 'animation_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 只使用第一个特征进行动画可视化
    feature_x = feature_columns[0]  # 面积
    X_1d = X[:, 0].reshape(-1, 1)  # 只使用面积

    # 特征标准化（使梯度下降更稳定）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_1d)

    print(f"使用特征: {feature_x}")
    print(f"最大迭代次数: {max_iterations}")

    # 使用梯度下降训练
    learning_rate = 0.01
    m, n = X_scaled.shape

    # 添加偏置项
    X_b = np.c_[np.ones((m, 1)), X_scaled]

    # 初始化参数
    theta = np.random.randn(2)
    theta_history = []
    cost_history = []

    # 计算最优解（正规方程）
    X_b_orig = np.c_[np.ones((X_1d.shape[0], 1)), X_1d]
    theta_optimal = np.linalg.inv(X_b_orig.T.dot(X_b_orig)).dot(X_b_orig.T).dot(y)

    # 梯度下降迭代
    for iteration in range(1, max_iterations + 1):
        # 计算预测值
        predictions = X_b.dot(theta)

        # 计算误差
        errors = predictions - y

        # 计算梯度
        gradients = (2/m) * X_b.T.dot(errors)

        # 更新参数
        theta = theta - learning_rate * gradients

        # 计算损失
        cost = np.mean(errors ** 2)
        theta_history.append(theta.copy())
        cost_history.append(cost)

        # 每2帧保存一次，或者第一帧和最后一帧
        if iteration == 1 or iteration == max_iterations or iteration % 2 == 0:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 左上图: 回归线演化
            ax1 = axes[0, 0]

            # 绘制散点图
            ax1.scatter(X_1d, y, alpha=0.6, s=50, c='#3498db',
                       edgecolors='black', linewidth=1, label='实际数据', zorder=3)

            # 绘制当前回归线
            x_range = np.linspace(X_1d.min(), X_1d.max(), 100)
            x_range_scaled = scaler.transform(x_range.reshape(-1, 1))
            x_range_b = np.c_[np.ones((len(x_range), 1)), x_range_scaled]
            y_pred = x_range_b.dot(theta)

            ax1.plot(x_range, y_pred, 'r-', linewidth=3, label='当前回归线', zorder=2)

            # 绘制最优回归线（虚线）
            y_pred_optimal = theta_optimal[0] + theta_optimal[1] * x_range
            ax1.plot(x_range, y_pred_optimal, 'g--', linewidth=2,
                    label='最优回归线', alpha=0.7, zorder=1)

            # 绘制残差线
            y_pred_all = X_b.dot(theta)
            for i in range(0, len(X_1d), 3):  # 每隔3个点画一条残差线
                ax1.plot([X_1d[i, 0], X_1d[i, 0]],
                        [y[i], y_pred_all[i]],
                        'orange', linewidth=1, alpha=0.5)

            ax1.set_xlabel(f'{feature_x} (平方米)', fontsize=12)
            ax1.set_ylabel('房价 (万元)', fontsize=12)
            ax1.set_title(f'回归线演化 - 迭代 {iteration}/{max_iterations}',
                         fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)

            # 右上图: 损失函数演化
            ax2 = axes[0, 1]

            ax2.plot(range(1, iteration + 1), cost_history, 'o-',
                    color='#e74c3c', linewidth=2, markersize=4)

            # 标注当前点
            ax2.scatter([iteration], [cost], s=200, c='#e74c3c',
                       marker='*', edgecolors='black', linewidth=2, zorder=5)

            ax2.set_xlim(0, max_iterations + 5)
            ax2.set_ylim(0, max(cost_history) * 1.1)
            ax2.set_xlabel('迭代次数', fontsize=12)
            ax2.set_ylabel('损失 (MSE)', fontsize=12)
            ax2.set_title('损失函数下降曲线', fontsize=13, fontweight='bold')
            ax2.grid(alpha=0.3)

            # 左下图: 参数演化
            ax3 = axes[1, 0]

            theta0_values = [t[0] for t in theta_history]
            theta1_values = [t[1] for t in theta_history]

            ax3.plot(range(1, iteration + 1), theta0_values, 'o-',
                    color='#3498db', linewidth=2, markersize=4,
                    label='theta_0 (截距)')
            ax3.plot(range(1, iteration + 1), theta1_values, 's-',
                    color='#e74c3c', linewidth=2, markersize=4,
                    label='theta_1 (斜率)')

            # 标注当前点
            ax3.scatter([iteration], [theta[0]], s=150, c='#3498db',
                       marker='*', edgecolors='black', linewidth=2, zorder=5)
            ax3.scatter([iteration], [theta[1]], s=150, c='#e74c3c',
                       marker='*', edgecolors='black', linewidth=2, zorder=5)

            ax3.set_xlim(0, max_iterations + 5)
            ax3.set_xlabel('迭代次数', fontsize=12)
            ax3.set_ylabel('参数值', fontsize=12)
            ax3.set_title('参数演化轨迹', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(alpha=0.3)

            # 右下图: 模型信息
            ax4 = axes[1, 1]
            ax4.axis('off')

            # 计算当前性能指标
            y_pred_all = X_b.dot(theta)
            r2 = r2_score(y, y_pred_all)
            mse = mean_squared_error(y, y_pred_all)

            info_text = f"""
            最小二乘回归优化过程

            迭代次数: {iteration} / {max_iterations}

            当前参数:
            - theta_0 (截距): {theta[0]:.4f}
            - theta_1 (斜率): {theta[1]:.4f}

            最优参数:
            - theta_0: {theta_optimal[0]:.4f}
            - theta_1: {theta_optimal[1]:.4f}

            模型性能:
            - 当前损失: {cost:.4f}
            - R^2: {r2:.4f}
            - MSE: {mse:.4f}

            损失变化:
            - 初始损失: {cost_history[0]:.4f}
            - 当前损失: {cost:.4f}
            - 降幅: {(cost_history[0] - cost)/cost_history[0]*100:.1f}%
            """

            ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('模型统计信息', fontsize=13, fontweight='bold')

            plt.suptitle('最小二乘回归 - 梯度下降优化过程',
                        fontsize=15, fontweight='bold')
            plt.tight_layout()

            # 保存帧
            frame_filename = os.path.join(frames_dir, f'frame_{iteration:03d}.png')
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            plt.close(fig)

            if iteration % 5 == 0 or iteration == max_iterations:
                print(f"  已生成 {iteration}/{max_iterations} 帧 "
                      f"(损失: {cost:.2f}, R^2: {r2:.4f})")

    print(f"\n所有帧已保存到: {frames_dir}/")

    # 生成GIF
    try:
        from PIL import Image
        print("\n正在生成GIF动画...")

        frames = []
        for iteration in range(1, max_iterations + 1):
            frame_filename = os.path.join(frames_dir, f'frame_{iteration:03d}.png')
            if os.path.exists(frame_filename):
                img = Image.open(frame_filename)
                frames.append(img)

        if frames:
            gif_path = 'least_squares_animation.gif'
            frames[0].save(gif_path,
                           save_all=True,
                           append_images=frames[1:],
                           duration=200,
                           loop=0)

            print(f"✅ GIF动画已保存为: {gif_path}")

    except ImportError:
        print("⚠️  PIL未安装，无法生成GIF")
        print("   安装方法: pip install Pillow")

    print("\n动画说明:")
    print("- 左上图: 回归线演化")
    print("  * 蓝色点: 实际数据")
    print("  * 红色线: 当前回归线")
    print("  * 绿色虚线: 最优回归线")
    print("  * 橙色线: 残差（误差）")
    print("- 右上图: 损失函数下降曲线")
    print("  * 展示MSE随迭代次数的变化")
    print("  * 红色星: 当前损失值")
    print("- 左下图: 参数演化轨迹")
    print("  * 蓝色线: 截距theta_0的变化")
    print("  * 红色线: 斜率theta_1的变化")
    print("- 右下图: 模型统计信息")
    print("  * 当前参数值和最优参数对比")
    print("  * 模型性能指标")
    print("\n观察要点:")
    print("- 回归线如何逐渐接近最优位置")
    print("- 损失函数如何单调下降")
    print("- 参数如何收敛到最优值")

def main():
    """主函数"""
    print("最小二乘回归算法示例 - 房价预测")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'least_squares_sample.csv'
    print(f"\n步骤1: 加载数据")
    print(f"{'='*70}")
    data = load_data(csv_path)
    print(f"数据加载成功: {csv_path}")
    print(f"数据形状: {data.shape}")
    print(f"\n数据前5行:")
    print(data.head())

    # 2. 准备数据
    print(f"\n步骤2: 准备数据")
    print(f"{'='*70}")
    X, y, feature_columns = prepare_data(data)
    print(f"特征列: {feature_columns}")
    print(f"样本数量: {len(X)}")
    print(f"目标变量: 房价")
    print(f"房价范围: {y.min():.2f} - {y.max():.2f} 万元")

    # 3. 划分数据集
    print(f"\n步骤3: 划分训练集和测试集")
    print(f"{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 4. 训练模型（正规方程）
    print(f"\n步骤4: 使用正规方程训练模型")
    print(f"{'='*70}")
    theta = normal_equation(X_train, y_train)
    print("模型训练完成!")
    print(f"回归系数:")
    print(f"  截距 (theta_0): {theta[0]:.4f}")
    for i, feat in enumerate(feature_columns):
        print(f"  {feat} (theta_{i+1}): {theta[i+1]:.4f}")

    # 5. 梯度下降对比
    print(f"\n步骤5: 使用梯度下降训练模型（对比）")
    print(f"{'='*70}")
    theta_gd, cost_history = gradient_descent(X_train, y_train,
                                              learning_rate=0.0001,
                                              n_iterations=1000)
    print(f"梯度下降完成!")
    print(f"最终损失: {cost_history[-1]:.4f}")
    print(f"回归系数 (梯度下降):")
    print(f"  截距: {theta_gd[0]:.4f}")
    for i, feat in enumerate(feature_columns):
        print(f"  {feat}: {theta_gd[i+1]:.4f}")

    # 6. 评估模型
    print(f"\n步骤6: 评估模型")
    print(f"{'='*70}")
    y_train_pred = predict(X_train, theta)
    y_test_pred = predict(X_test, theta)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"训练集:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R^2: {train_r2:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"\n测试集:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R^2: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.4f}")

    # 7. 生成可视化
    print(f"\n步骤7: 生成可视化图表")
    print(f"{'='*70}")
    visualize_regression_line(X_train, y_train, theta, feature_columns)
    visualize_residuals(y_test, y_test_pred)
    visualize_loss_surface(X_train, y_train, feature_columns)
    visualize_least_squares_geometry(X_train, y_train, theta, feature_columns)
    visualize_model_comparison(X_train, X_test, y_train, y_test, theta, feature_columns)

    # 8. 生成动画
    print(f"\n步骤8: 生成最小二乘回归动画")
    print(f"{'='*70}")
    create_animation(X_train, y_train, feature_columns, max_iterations=50)

    print(f"\n{'='*70}")
    print("最小二乘回归分析完成!")
    print("\n生成文件:")
    print("  - ls_regression_line.png (回归线)")
    print("  - ls_residuals.png (残差分析)")
    print("  - ls_loss_surface.png (损失曲面)")
    print("  - ls_geometry.png (几何意义)")
    print("  - ls_model_comparison.png (模型对比)")
    print("  - least_squares_animation.gif (优化过程动画)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
