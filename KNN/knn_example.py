"""
K近邻(KNN)算法最小实例
使用红酒品质数据作为示例: 根据化学成分预测红酒品质
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    print("\n品质分布:")
    print(data['品质'].value_counts())
    print(f"总样本数: {len(data)}")
    print(f"普通品质 (0): {sum(data['品质']==0)} 个")
    print(f"优质品质 (1): {sum(data['品质']==1)} 个")
    return data

def prepare_data(data):
    """准备训练数据"""
    feature_columns = ['含糖量(g/100ml)', '酸度(pH值)', '密度(g/cm3)', '酒精含量(%)']
    X = data[feature_columns].values
    y = data['品质'].values
    return X, y, feature_columns

def find_optimal_k(X_train, y_train, X_test, y_test, max_k=31):
    """寻找最优的K值"""
    k_range = range(1, max_k + 1, 2)  # 只测试奇数k值
    train_scores = []
    test_scores = []
    cv_scores = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n正在测试不同的K值...")

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)

        train_score = knn.score(X_train_scaled, y_train)
        test_score = knn.score(X_test_scaled, y_test)
        cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5).mean()

        train_scores.append(train_score)
        test_scores.append(test_score)
        cv_scores.append(cv_score)

        print(f"K={k:2d} | 训练准确率: {train_score*100:5.2f}% | "
              f"测试准确率: {test_score*100:5.2f}% | "
              f"交叉验证: {cv_score*100:5.2f}%")

    return k_range, train_scores, test_scores, cv_scores

def train_knn_models(X_train, y_train, k_values):
    """训练不同K值的KNN模型"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        models[k] = {
            'model': model,
            'scaler': scaler,
            'name': f'K={k}'
        }

    return models

def evaluate_model(model_info, X_test, y_test):
    """评估模型性能"""
    scaler = model_info['scaler']
    model = model_info['model']

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['普通品质', '优质品质'])

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

def visualize_k_selection(k_range, train_scores, test_scores, cv_scores):
    """可视化K值选择"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(k_range, train_scores, 'o-', color='#e74c3c',
           linewidth=2, markersize=6, label='训练集准确率')
    ax.plot(k_range, test_scores, 's-', color='#3498db',
           linewidth=2, markersize=6, label='测试集准确率')
    ax.plot(k_range, cv_scores, '^-', color='#2ecc71',
           linewidth=2, markersize=6, label='交叉验证准确率')

    # 标注最佳K值
    best_k_idx = np.argmax(test_scores)
    best_k = list(k_range)[best_k_idx]
    best_score = test_scores[best_k_idx]

    ax.axvline(x=best_k, color='orange', linestyle='--',
              linewidth=2, alpha=0.7, label=f'最佳K值: {best_k}')
    ax.scatter([best_k], [best_score], s=300, c='orange',
              marker='*', edgecolors='black', zorder=5,
              label=f'最佳测试准确率: {best_score*100:.2f}%')

    ax.set_xlabel('K值 (邻居数量)', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('KNN算法: K值选择与模型性能', fontsize=15)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(alpha=0.3)
    ax.set_xticks(k_range)
    ax.set_xticklabels(k_range, rotation=45)

    plt.tight_layout()
    plt.savefig('knn_k_selection.png', dpi=300, bbox_inches='tight')
    print("\nK值选择图已保存为 'knn_k_selection.png'")

def visualize_decision_boundary(X, y, feature_columns):
    """可视化不同K值的决策边界"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 只使用前两个特征进行2D可视化
    X_2d = X[:, :2]

    # 标准化
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # 创建网格
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    cmap_light = ListedColormap(['#FFB6C1', '#ADD8E6'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    k_values = [1, 3, 5, 11, 21, 31]

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # 训练KNN模型
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_2d_scaled, y)

        # 预测网格点
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

        # 绘制数据点
        ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y,
                  cmap=cmap_bold, edgecolors='black', s=40, alpha=0.8)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(feature_columns[0] + ' (标准化)', fontsize=10)
        ax.set_ylabel(feature_columns[1] + ' (标准化)', fontsize=10)
        ax.set_title(f'K={k} 的决策边界', fontsize=12)
        ax.grid(alpha=0.3)

    plt.suptitle('不同K值对决策边界的影响', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig('knn_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("决策边界图已保存为 'knn_decision_boundary.png'")

def visualize_knn_principle():
    """绘制KNN原理图"""
    fig = plt.figure(figsize=(16, 8))

    # 左图: KNN分类原理
    ax1 = plt.subplot(1, 2, 1)

    np.random.seed(42)
    # 类别A (红色)
    class_a = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 15)
    # 类别B (蓝色)
    class_b = np.random.multivariate_normal([7, 7], [[0.5, 0], [0, 0.5]], 15)

    # 新样本
    new_sample = np.array([[5, 5]])

    ax1.scatter(class_a[:, 0], class_a[:, 1], s=150, c='red',
               edgecolors='black', label='类别A (普通)', zorder=3, alpha=0.7)
    ax1.scatter(class_b[:, 0], class_b[:, 1], s=150, c='blue',
               edgecolors='black', label='类别B (优质)', zorder=3, alpha=0.7)

    # 新样本
    ax1.scatter(new_sample[:, 0], new_sample[:, 1], s=400, c='yellow',
               marker='*', edgecolors='black', linewidth=2,
               label='待分类样本', zorder=5)

    # K=5的邻居
    all_points = np.vstack([class_a, class_b])
    distances = np.sqrt(((all_points - new_sample) ** 2).sum(axis=1))
    nearest_indices = np.argsort(distances)[:5]

    for idx in nearest_indices:
        if idx < len(class_a):
            point = class_a[idx]
        else:
            point = class_b[idx - len(class_a)]
        ax1.plot([new_sample[0, 0], point[0]], [new_sample[0, 1], point[1]],
                'g--', linewidth=1.5, alpha=0.6)

    # 标注K=5圆圈
    circle = plt.Circle((new_sample[0, 0], new_sample[0, 1]),
                       distances[nearest_indices[-1]],
                       color='green', fill=False, linewidth=2,
                       linestyle='--', alpha=0.5, label='K=5 邻居范围')
    ax1.add_patch(circle)

    ax1.set_xlabel('含糖量 (标准化)', fontsize=12)
    ax1.set_ylabel('酸度 (标准化)', fontsize=12)
    ax1.set_title('KNN分类原理 (K=5)', fontsize=14)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 右图: 不同距离度量
    ax2 = plt.subplot(1, 2, 2)

    # 欧氏距离示例
    point1 = np.array([2, 2])
    point2 = np.array([6, 5])

    ax2.scatter([point1[0], point2[0]], [point1[1], point2[1]],
               s=300, c=['red', 'blue'], edgecolors='black', zorder=3)

    ax2.plot([point1[0], point2[0]], [point1[1], point2[1]],
            'g-', linewidth=3, label=f'欧氏距离 = {np.linalg.norm(point1 - point2):.2f}')

    # 添加坐标网格
    for i in range(8):
        ax2.axhline(y=i, color='gray', linestyle=':', alpha=0.3)
        ax2.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

    ax2.set_xlabel('特征1', fontsize=12)
    ax2.set_ylabel('特征2', fontsize=12)
    ax2.set_title('距离度量示例: 欧氏距离', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)

    # 添加距离公式
    formula_text = r'$d(p_1, p_2) = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$'
    ax2.text(0.5, 0.5, formula_text, fontsize=16, ha='center', va='center',
            transform=ax2.transAxes, bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('knn_principle.png', dpi=300, bbox_inches='tight')
    print("KNN原理图已保存为 'knn_principle.png'")

def visualize_distances_comparison(X_train, y_train, X_test, y_test):
    """可视化不同距离度量的对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    distance_names = ['欧氏距离', '曼哈顿距离', '切比雪夫距离', '闵可夫斯基距离']
    accuracies = []

    for metric in distance_metrics:
        if metric == 'minkowski':
            knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
        else:
            knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

        knn.fit(X_train_scaled, y_train)
        score = knn.score(X_test_scaled, y_test)
        accuracies.append(score * 100)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = ax.bar(distance_names, accuracies, color=colors,
                  edgecolor='black', alpha=0.8)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold')

    ax.set_ylabel('准确率 (%)', fontsize=13)
    ax.set_title('不同距离度量对KNN性能的影响', fontsize=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([70, 100])

    plt.tight_layout()
    plt.savefig('knn_distance_metrics.png', dpi=300, bbox_inches='tight')
    print("距离度量对比图已保存为 'knn_distance_metrics.png'")

def visualize_feature_distributions(data, feature_columns):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        class_0 = data[data['品质'] == 0][feature]
        class_1 = data[data['品质'] == 1][feature]

        ax.hist(class_0, bins=15, color='#e74c3c', alpha=0.6,
               label='普通品质', edgecolor='black')
        ax.hist(class_1, bins=15, color='#3498db', alpha=0.6,
               label='优质品质', edgecolor='black')

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{feature} 分布', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('knn_feature_distribution.png', dpi=300, bbox_inches='tight')
    print("特征分布图已保存为 'knn_feature_distribution.png'")

def main():
    """主函数"""
    print("K近邻(KNN)算法示例 - 红酒品质预测")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'knn_sample.csv'
    data = load_data(csv_path)

    # 2. 准备数据
    X, y, feature_columns = prepare_data(data)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n数据集划分:")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")

    # 4. 寻找最优K值
    print(f"\n{'='*70}")
    print("步骤1: 寻找最优K值")
    print(f"{'='*70}")
    k_range, train_scores, test_scores, cv_scores = find_optimal_k(
        X_train, y_train, X_test, y_test
    )

    # 5. 训练不同K值的模型
    print(f"\n{'='*70}")
    print("步骤2: 训练选定的KNN模型")
    print(f"{'='*70}")

    # 选择几个代表性的K值
    selected_ks = [1, 5, 11, 21]
    models = train_knn_models(X_train, y_train, selected_ks)

    # 6. 评估模型
    print(f"\n{'='*70}")
    print("步骤3: 模型评估")
    print(f"{'='*70}")

    results = {}
    for k in selected_ks:
        result = evaluate_model(models[k], X_test, y_test)
        results[k] = result
        print(f"\nK={k} 模型:")
        print(f"  准确率: {result['accuracy']*100:.2f}%")
        print(f"  AUC值: {result['roc_auc']:.4f}")

    # 找出最佳K值
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_k]

    print(f"\n最佳K值: K={best_k}")
    print(f"准确率: {best_result['accuracy']*100:.2f}%")
    print(f"\n混淆矩阵:")
    print(best_result['conf_matrix'])
    print(f"\n分类报告:")
    print(best_result['report'])

    # 7. 示例预测
    print(f"\n{'='*70}")
    print("步骤4: 示例预测")
    print(f"{'='*70}")

    sample_wines = np.array([
        [2.5, 3.0, 0.9945, 9.0],   # 普通品质
        [8.5, 3.0, 1.0045, 12.5],  # 优质品质
        [5.5, 2.9, 0.9998, 10.5]   # 边界样本
    ])

    scaler = models[best_k]['scaler']
    sample_scaled = scaler.transform(sample_wines)
    sample_predictions = models[best_k]['model'].predict(sample_scaled)
    sample_probabilities = models[best_k]['model'].predict_proba(sample_scaled)

    # 获取最近的邻居
    distances, indices = models[best_k]['model'].kneighbors(sample_scaled)

    descriptions = [
        "低含糖量红酒 (预计普通)",
        "高含糖量红酒 (预计优质)",
        "中等含糖量红酒 (边界情况)"
    ]

    for i, (desc, pred, prob, dist, idx) in enumerate(zip(
        descriptions, sample_predictions, sample_probabilities,
        distances, indices), 1):
        quality = "优质品质" if pred == 1 else "普通品质"
        print(f"\n样本{i} ({desc}):")
        print(f"  含糖量: {sample_wines[i-1][0]:.1f} g/100ml")
        print(f"  酸度: {sample_wines[i-1][1]:.1f}")
        print(f"  密度: {sample_wines[i-1][2]:.4f} g/cm³")
        print(f"  酒精含量: {sample_wines[i-1][3]:.1f}%")
        print(f"  预测品质: {quality}")
        print(f"  普通品质概率: {prob[0]*100:.2f}%")
        print(f"  优质品质概率: {prob[1]*100:.2f}%")
        print(f"  最近{best_k}个邻居的平均距离: {dist.mean():.4f}")

    # 8. 可视化
    print(f"\n{'='*70}")
    print("步骤5: 生成可视化图表")
    print(f"{'='*70}")

    visualize_k_selection(k_range, train_scores, test_scores, cv_scores)
    visualize_decision_boundary(X, y, feature_columns)
    visualize_knn_principle()
    visualize_distances_comparison(X_train, y_train, X_test, y_test)
    visualize_feature_distributions(data, feature_columns)

    print(f"\n{'='*70}")
    print("KNN分析完成!")
    print("\n生成文件:")
    print("  - knn_k_selection.png (K值选择曲线)")
    print("  - knn_decision_boundary.png (决策边界可视化)")
    print("  - knn_principle.png (KNN原理图)")
    print("  - knn_distance_metrics.png (距离度量对比)")
    print("  - knn_feature_distribution.png (特征分布)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
