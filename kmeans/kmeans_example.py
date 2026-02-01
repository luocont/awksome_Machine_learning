"""
K-Means聚类算法最小实例
使用客户数据作为示例: 根据消费行为进行客户分群
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path):
    """加载CSV数据"""
    data = pd.read_csv(csv_path, encoding='utf-8')
    print("=" * 70)
    print("数据预览:")
    print(data.head(10))
    print("\n数据统计信息:")
    print(data.describe())
    print(f"\n总样本数: {len(data)}")
    print(f"特征数量: {len(data.columns)}")
    return data

def prepare_data(data):
    """准备数据"""
    feature_columns = ['年龄', '年收入(万元)', '年消费(万元)', '购物频率(次/月)', '在线时长(小时/周)']
    X = data[feature_columns].values

    # 标准化数据(K-Means对尺度敏感)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, feature_columns, scaler

def find_optimal_k(X_scaled, max_k=10):
    """使用多种方法寻找最优K值"""
    print("\n" + "=" * 70)
    print("寻找最优K值...")
    print("=" * 70)

    k_range = range(2, max_k + 1)
    inertias = []          # 肘部法则
    silhouette_scores = [] # 轮廓系数
    calinski_scores = []   # Calinski-Harabasz指数
    davies_scores = []     # Davies-Bouldin指数

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # 计算各种评估指标
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
        davies_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

        print(f"K={k:2d} | 惯性: {inertias[-1]:8.2f} | "
              f"轮廓系数: {silhouette_scores[-1]:.4f} | "
              f"CH指数: {calinski_scores[-1]:7.2f} | "
              f"DB指数: {davies_scores[-1]:.4f}")

    return k_range, inertias, silhouette_scores, calinski_scores, davies_scores

def train_kmeans(X_scaled, n_clusters):
    """训练K-Means模型"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans

def analyze_clusters(kmeans, data, X_scaled, scaler):
    """分析聚类结果"""
    print("\n" + "=" * 70)
    print("聚类分析结果")
    print("=" * 70)

    # 添加聚类标签到原始数据
    data_with_clusters = data.copy()
    data_with_clusters['聚类'] = kmeans.labels_

    # 统计每个聚类的样本数
    print("\n各聚类样本数:")
    cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = count / len(kmeans.labels_) * 100
        print(f"  聚类 {cluster_id}: {count:3d} 个样本 ({percentage:5.2f}%)")

    # 计算每个聚类的中心点(还原到原始尺度)
    print("\n各聚类中心点(原始尺度):")
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    feature_names = ['年龄', '年收入', '年消费', '购物频率', '在线时长']
    for i, centroid in enumerate(centroids_original):
        print(f"\n聚类 {i} 中心特征:")
        for name, value in zip(feature_names, centroid):
            print(f"  {name:12s}: {value:7.2f}")

    # 每个聚类的统计信息
    print("\n" + "=" * 70)
    print("各聚类详细统计:")
    print("=" * 70)

    for cluster_id in range(kmeans.n_clusters):
        cluster_data = data_with_clusters[data_with_clusters['聚类'] == cluster_id]
        print(f"\n--- 聚类 {cluster_id} ---")
        print(cluster_data.describe())

    return data_with_clusters, centroids_original

def visualize_k_selection(k_range, inertias, silhouette_scores,
                         calinski_scores, davies_scores):
    """可视化K值选择"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 肘部法则
    ax1 = axes[0, 0]
    ax1.plot(k_range, inertias, 'o-', color='#e74c3c',
            linewidth=2, markersize=8)
    ax1.set_xlabel('聚类数量 K', fontsize=12)
    ax1.set_ylabel('惯性(Inertia)', fontsize=12)
    ax1.set_title('肘部法则: 寻找拐点', fontsize=13)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_range)

    # 标注可能的肘部
    if len(k_range) > 2:
        # 计算曲率变化最大的点
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 1
        if elbow_idx < len(k_range):
            ax1.plot(k_range[elbow_idx], inertias[elbow_idx],
                    'y*', markersize=20, label=f'可能的肘部 K={k_range[elbow_idx]}')
            ax1.legend()

    # 子图2: 轮廓系数
    ax2 = axes[0, 1]
    ax2.plot(k_range, silhouette_scores, 's-', color='#3498db',
            linewidth=2, markersize=8)
    best_silhouette_idx = np.argmax(silhouette_scores)
    ax2.plot(k_range[best_silhouette_idx], silhouette_scores[best_silhouette_idx],
            'g*', markersize=20, label=f'最佳 K={k_range[best_silhouette_idx]}')
    ax2.set_xlabel('聚类数量 K', fontsize=12)
    ax2.set_ylabel('轮廓系数 (Silhouette Score)', fontsize=12)
    ax2.set_title('轮廓系数: 越大越好', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(k_range)

    # 子图3: Calinski-Harabasz指数
    ax3 = axes[1, 0]
    ax3.plot(k_range, calinski_scores, '^-', color='#2ecc71',
            linewidth=2, markersize=8)
    best_ch_idx = np.argmax(calinski_scores)
    ax3.plot(k_range[best_ch_idx], calinski_scores[best_ch_idx],
            'g*', markersize=20, label=f'最佳 K={k_range[best_ch_idx]}')
    ax3.set_xlabel('聚类数量 K', fontsize=12)
    ax3.set_ylabel('CH指数 (越大越好)', fontsize=12)
    ax3.set_title('Calinski-Harabasz指数', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xticks(k_range)

    # 子图4: Davies-Bouldin指数
    ax4 = axes[1, 1]
    ax4.plot(k_range, davies_scores, 'd-', color='#f39c12',
            linewidth=2, markersize=8)
    best_db_idx = np.argmin(davies_scores)
    ax4.plot(k_range[best_db_idx], davies_scores[best_db_idx],
            'g*', markersize=20, label=f'最佳 K={k_range[best_db_idx]}')
    ax4.set_xlabel('聚类数量 K', fontsize=12)
    ax4.set_ylabel('DB指数 (越小越好)', fontsize=12)
    ax4.set_title('Davies-Bouldin指数', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_xticks(k_range)

    plt.tight_layout()
    plt.savefig('kmeans_k_selection.png', dpi=300, bbox_inches='tight')
    print("\nK值选择图已保存为 'kmeans_k_selection.png'")

def visualize_clusters_2d(X_scaled, labels, centroids):
    """2D可视化聚类结果(使用PCA降维)"""
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 定义颜色
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # 绘制数据点
    for i in range(len(np.unique(labels))):
        cluster_points = X_pca[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  c=colors[i % len(colors)], label=f'聚类 {i}',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # 绘制中心点
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
              c='black', marker='X', s=300, edgecolors='yellow',
              linewidth=2, label='聚类中心', zorder=5)

    ax.set_xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]*100:.1f}%)',
                 fontsize=12)
    ax.set_ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]*100:.1f}%)',
                 fontsize=12)
    ax.set_title('K-Means聚类结果 (PCA降维可视化)', fontsize=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('kmeans_clusters_2d.png', dpi=300, bbox_inches='tight')
    print("2D聚类图已保存为 'kmeans_clusters_2d.png'")

def visualize_clusters_3d(X_scaled, labels, centroids):
    """3D可视化聚类结果"""
    # 使用PCA降维到3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 定义颜色
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # 绘制数据点
    for i in range(len(np.unique(labels))):
        cluster_points = X_pca[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  cluster_points[:, 2], c=colors[i % len(colors)],
                  label=f'聚类 {i}', alpha=0.6, s=50,
                  edgecolors='black', linewidth=0.5)

    # 绘制中心点
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
              centroids_pca[:, 2], c='black', marker='X',
              s=300, edgecolors='yellow', linewidth=3,
              label='聚类中心', zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=11)
    ax.set_title('K-Means聚类结果 (3D可视化)', fontsize=15, pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('kmeans_clusters_3d.png', dpi=300, bbox_inches='tight')
    print("3D聚类图已保存为 'kmeans_clusters_3d.png'")

def visualize_feature_distributions(data_with_clusters, feature_columns):
    """可视化各特征在不同聚类中的分布"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        # 为每个聚类绘制箱线图
        cluster_data = []
        cluster_labels = []
        for cluster_id in sorted(data_with_clusters['聚类'].unique()):
            cluster_data.append(data_with_clusters[
                data_with_clusters['聚类'] == cluster_id][feature])
            cluster_labels.append(f'聚类{cluster_id}')

        bp = ax.boxplot(cluster_data, tick_labels=cluster_labels,
                       patch_artist=True, showmeans=True)

        # 为每个箱线图设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(feature, fontsize=11)
        ax.set_title(f'{feature} 分布', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    # 移除多余的子图
    if len(feature_columns) < len(axes):
        for idx in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('kmeans_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("特征分布图已保存为 'kmeans_feature_distributions.png'")

def visualize_kmeans_principle():
    """绘制K-Means原理图"""
    fig = plt.figure(figsize=(16, 8))

    # 左图: K-Means迭代过程示意
    ax1 = plt.subplot(1, 2, 1)

    np.random.seed(42)
    # 生成3个聚类的数据
    cluster1 = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 30)
    cluster2 = np.random.multivariate_normal([7, 7], [[0.5, 0], [0, 0.5]], 30)
    cluster3 = np.random.multivariate_normal([3, 7], [[0.5, 0], [0, 0.5]], 30)

    # 绘制数据点
    ax1.scatter(cluster1[:, 0], cluster1[:, 1], c='red',
               alpha=0.6, s=80, edgecolors='black', linewidth=0.5, label='聚类1')
    ax1.scatter(cluster2[:, 0], cluster2[:, 1], c='blue',
               alpha=0.6, s=80, edgecolors='black', linewidth=0.5, label='聚类2')
    ax1.scatter(cluster3[:, 0], cluster3[:, 1], c='green',
               alpha=0.6, s=80, edgecolors='black', linewidth=0.5, label='聚类3')

    # 绘制中心点
    centroids = np.array([[3, 3], [7, 7], [3, 7]])
    ax1.scatter(centroids[:, 0], centroids[:, 1],
               c='black', marker='X', s=400,
               edgecolors='yellow', linewidth=2, label='聚类中心', zorder=5)

    # 绘制一些点到中心的连线
    for i, (cluster, centroid, color) in enumerate(
        [(cluster1, centroids[0], 'red'),
         (cluster2, centroids[1], 'blue'),
         (cluster3, centroids[2], 'green')]):

        # 随机选择几个点连线
        for _ in range(3):
            idx = np.random.randint(len(cluster))
            point = cluster[idx]
            ax1.plot([point[0], centroid[0]], [point[1], centroid[1]],
                    color=color, linestyle='--', alpha=0.4, linewidth=1)

    ax1.set_xlabel('特征1', fontsize=13)
    ax1.set_ylabel('特征2', fontsize=13)
    ax1.set_title('K-Means聚类示意', fontsize=15)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 右图: K-Means算法流程
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    steps = [
        "1. 初始化",
        "   随机选择K个点作为初始聚类中心",
        "",
        "2. 分配样本",
        "   计算每个样本到各中心的距离",
        "   将样本分配到最近的中心",
        "",
        "3. 更新中心",
        "   计算每个聚类的新的中心点",
        "   (该聚类所有点的均值)",
        "",
        "4. 迭代",
        "   重复步骤2-3",
        "   直到中心点不再变化或达到最大迭代次数"
    ]

    y_position = 0.95
    for line in steps:
        if line.startswith(("1.", "2.", "3.", "4.")):
            ax2.text(0.1, y_position, line, fontsize=14,
                    fontweight='bold', color='#e74c3c',
                    transform=ax2.transAxes)
            y_position -= 0.08
        elif line.strip():
            ax2.text(0.15, y_position, line, fontsize=12,
                    transform=ax2.transAxes)
            y_position -= 0.05
        else:
            y_position -= 0.03

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('kmeans_principle.png', dpi=300, bbox_inches='tight')
    print("K-Means原理图已保存为 'kmeans_principle.png'")

def visualize_radar_chart(centroids_original, feature_columns):
    """绘制各聚类的雷达图"""
    # 只取前4个特征用于雷达图
    features_for_radar = feature_columns[:4]
    centroids_for_radar = centroids_original[:, :4]

    # 标准化到0-1范围
    centroids_norm = (centroids_for_radar - centroids_for_radar.min(axis=0)) / \
                     (centroids_for_radar.max(axis=0) - centroids_for_radar.min(axis=0))

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(features_for_radar), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (centroid, color) in enumerate(zip(centroids_norm, colors)):
        values = centroid.tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=f'聚类 {i}',
               color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_for_radar, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('各聚类特征雷达图', fontsize=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('kmeans_radar_chart.png', dpi=300, bbox_inches='tight')
    print("雷达图已保存为 'kmeans_radar_chart.png'")

def create_animation(X_scaled, n_clusters=4):
    """创建K-Means动画 - 逐步添加数据点展示聚类演化"""
    print("\n" + "=" * 70)
    print("开始生成K-Means动画...")
    print("=" * 70)

    # 创建保存帧的目录
    frames_dir = 'animation_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 使用PCA降维到2D便于可视化
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    print(f"使用 {n_clusters} 个聚类")
    print(f"总共有 {len(X_scaled)} 个数据点")

    # 定义颜色
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # 打乱数据顺序
    indices = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[indices]
    X_2d_shuffled = X_2d[indices]

    # 从少量数据点开始，逐步添加
    start_points = max(n_clusters * 2, 10)  # 至少是聚类数的2倍

    for n_points in range(start_points, len(X_shuffled) + 1):
        # 取前n_points个数据点
        x_current = X_shuffled[:n_points]
        x_2d_current = X_2d_shuffled[:n_points]

        # 训练K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(x_current)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers_2d = pca.transform(centers)

        # 计算评估指标
        inertia = kmeans.inertia_
        silhouette = silhouette_score(x_current, labels)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 左图: 2D聚类结果
        for i in range(n_clusters):
            cluster_points = x_2d_current[labels == i]
            ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=colors[i % len(colors)], label=f'聚类 {i}',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # 绘制中心点
        ax1.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c='black', marker='X', s=400, edgecolors='yellow',
                   linewidth=3, label='聚类中心', zorder=5)

        # 绘制待添加的点(灰色)
        if n_points < len(X_shuffled):
            remaining_2d = X_2d_shuffled[n_points:]
            ax1.scatter(remaining_2d[:, 0], remaining_2d[:, 1],
                       c='lightgray', s=30, alpha=0.3, edgecolors='gray',
                       label='待添加点', zorder=1)

        ax1.set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax1.set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax1.set_title(f'K-Means聚类演化 - {n_points} 个数据点', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(alpha=0.3)

        # 右图: 惯性变化曲线
        if hasattr(create_animation, 'inertia_history'):
            create_animation.inertia_history.append(inertia)
            create_animation.points_history.append(n_points)
        else:
            create_animation.inertia_history = [inertia]
            create_animation.points_history = [n_points]

        ax2.plot(create_animation.points_history, create_animation.inertia_history,
                'o-', color='#e74c3c', linewidth=2, markersize=8)
        ax2.axhline(y=inertia, color='blue', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'当前惯性: {inertia:.2f}')

        # 标注当前点
        ax2.scatter([n_points], [inertia], s=200, c='red', marker='*',
                   edgecolors='black', linewidth=2, zorder=5)

        ax2.set_xlabel('数据点数量', fontsize=12)
        ax2.set_ylabel('惯性 (Inertia)', fontsize=12)
        ax2.set_title('惯性随数据增加的变化', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        # 保存帧
        frame_filename = os.path.join(frames_dir, f'frame_{n_points:03d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if n_points % 10 == 0 or n_points == len(X_shuffled):
            print(f"  已生成 {n_points}/{len(X_shuffled)} 帧 (惯性: {inertia:.2f})")

    print(f"\n所有帧已保存到: {frames_dir}/")

    # 生成GIF
    try:
        from PIL import Image
        print("\n正在生成GIF动画...")

        frames = []
        for n_points in range(start_points, len(X_shuffled) + 1):
            frame_filename = os.path.join(frames_dir, f'frame_{n_points:03d}.png')
            img = Image.open(frame_filename)
            frames.append(img)

        gif_path = 'kmeans_animation.gif'
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=300,
                       loop=0)

        print(f"✅ GIF动画已保存为: {gif_path}")

    except ImportError:
        print("⚠️  PIL未安装，无法生成GIF")
        print("   安装方法: pip install Pillow")

    # 清理临时数据
    if hasattr(create_animation, 'inertia_history'):
        delattr(create_animation, 'inertia_history')
    if hasattr(create_animation, 'points_history'):
        delattr(create_animation, 'points_history')

    print("\n动画说明:")
    print("- 左图: K-Means聚类结果演化")
    print("  * 彩色散点: 不同聚类的样本")
    print("  * 黑色X: 聚类中心")
    print("  * 灰色点: 待添加的数据点")
    print("- 右图: 惯性随数据增加的变化曲线")
    print("  * 惯性越小表示聚类越紧密")
    print("  * 观察惯性如何随着数据增加而变化")
    print("- 每帧增加新数据点，展示聚类如何随数据增长而演化")

def main():
    """主函数"""
    print("K-Means聚类算法示例 - 客户分群")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'kmeans_sample.csv'
    data = load_data(csv_path)

    # 2. 准备数据
    X, X_scaled, feature_columns, scaler = prepare_data(data)

    # 3. 寻找最优K值
    k_range, inertias, silhouette_scores, calinski_scores, davies_scores = \
        find_optimal_k(X_scaled, max_k=10)

    # 推荐K值(综合多个指标)
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    best_ch_k = k_range[np.argmax(calinski_scores)]
    best_db_k = k_range[np.argmin(davies_scores)]

    print("\n" + "=" * 70)
    print("最优K值推荐:")
    print(f"  轮廓系数推荐: K={best_silhouette_k}")
    print(f"  CH指数推荐: K={best_ch_k}")
    print(f"  DB指数推荐: K={best_db_k}")
    print("=" * 70)

    # 选择K值(这里选择轮廓系数最大的)
    optimal_k = best_silhouette_k
    print(f"\n选择 K={optimal_k} 进行聚类分析")

    # 4. 训练K-Means模型
    print("\n" + "=" * 70)
    print("训练K-Means模型...")
    print("=" * 70)
    kmeans = train_kmeans(X_scaled, optimal_k)

    # 5. 分析聚类结果
    data_with_clusters, centroids_original = analyze_clusters(
        kmeans, data, X_scaled, scaler
    )

    # 6. 计算最终评估指标
    print("\n" + "=" * 70)
    print("聚类评估指标:")
    print("=" * 70)
    final_silhouette = silhouette_score(X_scaled, kmeans.labels_)
    final_calinski = calinski_harabasz_score(X_scaled, kmeans.labels_)
    final_davies = davies_bouldin_score(X_scaled, kmeans.labels_)

    print(f"轮廓系数: {final_silhouette:.4f} (越接近1越好)")
    print(f"CH指数: {final_calinski:.2f} (越大越好)")
    print(f"DB指数: {final_davies:.4f} (越小越好)")
    print(f"惯性: {kmeans.inertia_:.2f}")

    # 7. 聚类解释
    print("\n" + "=" * 70)
    print("聚类解释:")
    print("=" * 70)

    cluster_descriptions = {
        0: "低收入年轻群体 - 学生/刚毕业",
        1: "中收入中年群体 - 稳定职业",
        2: "高收入中年群体 - 企业高管",
        3: "高收入青年群体 - 创业者/高技能人才"
    }

    if optimal_k <= len(cluster_descriptions):
        for i in range(optimal_k):
            print(f"\n聚类 {i}: {cluster_descriptions.get(i, '待分析')}")

    # 8. 可视化
    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)

    visualize_k_selection(k_range, inertias, silhouette_scores,
                         calinski_scores, davies_scores)
    visualize_clusters_2d(X_scaled, kmeans.labels_, kmeans.cluster_centers_)
    visualize_clusters_3d(X_scaled, kmeans.labels_, kmeans.cluster_centers_)
    visualize_feature_distributions(data_with_clusters, feature_columns)
    visualize_kmeans_principle()
    visualize_radar_chart(centroids_original, feature_columns)

    # 9. 生成动画
    create_animation(X_scaled, optimal_k)

    print("\n" + "=" * 70)
    print("K-Means聚类分析完成!")
    print("\n生成文件:")
    print("  - kmeans_k_selection.png (K值选择)")
    print("  - kmeans_clusters_2d.png (2D聚类可视化)")
    print("  - kmeans_clusters_3d.png (3D聚类可视化)")
    print("  - kmeans_feature_distributions.png (特征分布)")
    print("  - kmeans_principle.png (K-Means原理)")
    print("  - kmeans_radar_chart.png (聚类特征雷达图)")
    print("  - kmeans_animation.gif (迭代过程动画)")
    print("=" * 70)

if __name__ == "__main__":
    main()
