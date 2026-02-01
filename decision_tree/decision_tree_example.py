"""
决策树算法最小实例
使用贷款审批数据作为示例: 根据申请人信息预测是否违约
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
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
    print("=" * 70)
    print("数据预览:")
    print(data.head(10))
    print("\n数据统计信息:")
    print(data.describe())
    print("\n违约情况统计:")
    print(data['违约记录'].value_counts())
    print(f"总样本数: {len(data)}")
    print(f"正常客户 (0): {sum(data['违约记录']==0)} 个")
    print(f"违约客户 (1): {sum(data['违约记录']==1)} 个")
    print(f"违约率: {data['违约记录'].mean() * 100:.2f}%")
    return data

def prepare_data(data):
    """准备数据"""
    feature_columns = ['年龄', '收入(万元)', '工作年限', '信用卡额度(万元)', '负债率']
    X = data[feature_columns].values
    y = data['违约记录'].values
    return X, y, feature_columns

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2):
    """训练决策树模型"""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=1,
        random_state=42,
        criterion='gini'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['正常', '违约'])

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

def visualize_tree_structure(model, feature_columns, class_names):
    """可视化决策树结构"""
    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(model,
             feature_names=feature_columns,
             class_names=class_names,
             filled=True,
             rounded=True,
             fontsize=10,
             impurity=False,
             ax=ax)

    ax.set_title('决策树结构图', fontsize=18, pad=20)

    plt.tight_layout()
    plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
    print("\n决策树结构图已保存为 'decision_tree_structure.png'")

def print_tree_rules(model, feature_columns):
    """打印决策树规则"""
    tree_rules = export_text(model,
                             feature_names=feature_columns,
                             show_weights=True)

    print("\n" + "=" * 70)
    print("决策树规则:")
    print("=" * 70)
    print(tree_rules)
    print("=" * 70)

def visualize_feature_importance(model, feature_columns):
    """可视化特征重要性"""
    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))

    # 按重要性排序
    indices = np.argsort(importance)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))

    bars = ax.barh(range(len(importance)), importance[indices],
                   color=colors[indices], edgecolor='black', alpha=0.8)

    # 添加数值标签
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{importance[idx]:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('特征重要性', fontsize=13)
    ax.set_title('决策树 - 特征重要性分析', fontsize=15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n特征重要性图已保存为 'decision_tree_feature_importance.png'")

def visualize_roc_curve(metrics):
    """可视化ROC曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics['fpr'], metrics['tpr'],
           color='#3498db', linewidth=2.5,
           label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.4f})')

    ax.plot([0, 1], [0, 1], color='red', linestyle='--',
           linewidth=2, label='随机分类器')

    # 标注最佳阈值点
    optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
    ax.plot(metrics['fpr'][optimal_idx],
           metrics['tpr'][optimal_idx],
           'go', markersize=10, label='最佳阈值点')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 曲线', fontsize=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_tree_roc_curve.png', dpi=300, bbox_inches='tight')
    print("\nROC曲线图已保存为 'decision_tree_roc_curve.png'")

def visualize_confusion_matrix(conf_matrix):
    """可视化混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black",
                   fontsize=16)

    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['正常', '违约'])
    ax.set_yticklabels(['正常', '违约'])

    plt.tight_layout()
    plt.savefig('decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n混淆矩阵图已保存为 'decision_tree_confusion_matrix.png'")

def visualize_depth_impact(X_train, y_train, X_test, y_test):
    """可视化不同深度对模型性能的影响"""
    max_depths = range(2, 16)
    train_scores = []
    test_scores = []
    cv_scores = []

    print("\n正在测试不同树深度的性能...")

    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        train_scores.append(train_score)
        test_scores.append(test_score)
        cv_scores.append(cv_score)

        print(f"深度={depth:2d} | 训练: {train_score*100:5.2f}% | "
             f"测试: {test_score*100:5.2f}% | 交叉验证: {cv_score*100:5.2f}%")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(max_depths, train_scores, 'o-', color='#e74c3c',
           linewidth=2, markersize=6, label='训练集准确率')
    ax.plot(max_depths, test_scores, 's-', color='#3498db',
           linewidth=2, markersize=6, label='测试集准确率')
    ax.plot(max_depths, cv_scores, '^-', color='#2ecc71',
           linewidth=2, markersize=6, label='交叉验证准确率')

    # 标注最佳深度
    best_depth_idx = np.argmax(test_scores)
    best_depth = list(max_depths)[best_depth_idx]
    best_score = test_scores[best_depth_idx]

    ax.axvline(x=best_depth, color='orange', linestyle='--',
              linewidth=2, alpha=0.7, label=f'最佳深度: {best_depth}')
    ax.scatter([best_depth], [best_score], s=300, c='orange',
              marker='*', edgecolors='black', zorder=5,
              label=f'最佳测试准确率: {best_score*100:.2f}%')

    ax.set_xlabel('树的最大深度', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('决策树: 深度对性能的影响', fontsize=15)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(alpha=0.3)
    ax.set_xticks(max_depths)

    plt.tight_layout()
    plt.savefig('decision_tree_depth_analysis.png', dpi=300, bbox_inches='tight')
    print("\n深度分析图已保存为 'decision_tree_depth_analysis.png'")

    return best_depth

def visualize_decision_boundary(model, X, y, feature_columns):
    """可视化决策边界(使用前两个特征)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 只使用前两个特征进行2D可视化
    X_2d = X[:, :2]

    # 训练2D模型
    model_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_2d.fit(X_2d, y)

    # 创建网格
    x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
    y_min, y_max = X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    # 左图: 决策边界
    ax1 = axes[0]

    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFB6C1', '#ADD8E6'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    ax1.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold,
               edgecolors='black', s=60, alpha=0.8)

    ax1.set_xlabel(feature_columns[0], fontsize=12)
    ax1.set_ylabel(feature_columns[1], fontsize=12)
    ax1.set_title('决策边界可视化 (深度=3)', fontsize=14)
    ax1.grid(alpha=0.3)

    # 右图: 决策树路径
    ax2 = axes[1]

    # 绘制决策树的分割线
    def draw_tree_boundary(node, x_min, x_max, y_min, y_max):
        if model_2d.tree_.feature[node[0]] != -2:  # 不是叶子节点
            feature = model_2d.tree_.feature[node[0]]
            threshold = model_2d.tree_.threshold[node[0]]

            if feature == 0:  # 年龄
                ax2.axvline(x=threshold, ymin=y_min, ymax=y_max,
                           color='black', linestyle='--', linewidth=1.5, alpha=0.5)
                draw_tree_boundary((node[1],), x_min, threshold, y_min, y_max)
                draw_tree_boundary((node[2],), threshold, x_max, y_min, y_max)
            else:  # 收入
                ax2.axhline(y=threshold, xmin=x_min, xmax=x_max,
                           color='black', linestyle='--', linewidth=1.5, alpha=0.5)
                draw_tree_boundary((node[1],), x_min, x_max, y_min, threshold)
                draw_tree_boundary((node[2],), x_min, x_max, threshold, y_max)

    # 绘制数据点
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold,
               edgecolors='black', s=60, alpha=0.8)

    ax2.set_xlabel(feature_columns[0], fontsize=12)
    ax2.set_ylabel(feature_columns[1], fontsize=12)
    ax2.set_title('决策树分割规则', fontsize=14)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_tree_boundary.png', dpi=300, bbox_inches='tight')
    print("\n决策边界图已保存为 'decision_tree_boundary.png'")

def visualize_tree_principle():
    """绘制决策树原理图"""
    fig = plt.figure(figsize=(16, 10))

    # 决策树示意图
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')

    # 绘制决策树
    nodes = [
        {'pos': (0.5, 0.95), 'text': '负债率 ≤ 0.35?\n基尼系数: 0.48\n样本: 100', 'level': 0},
        {'pos': (0.25, 0.75), 'text': '收入 ≤ 15万?\n基尼系数: 0.32\n样本: 60', 'level': 1, 'parent': 0, 'edge': '是'},
        {'pos': (0.75, 0.75), 'text': '工作年限 ≤ 5年?\n基尼系数: 0.17\n样本: 40', 'level': 1, 'parent': 0, 'edge': '否'},
        {'pos': (0.125, 0.55), 'text': '违约\n基尼系数: 0.0\n样本: 25', 'level': 2, 'parent': 1, 'edge': '是'},
        {'pos': (0.375, 0.55), 'text': '正常\n基尼系数: 0.1\n样本: 35', 'level': 2, 'parent': 1, 'edge': '否'},
        {'pos': (0.625, 0.55), 'text': '正常\n基尼系数: 0.05\n样本: 30', 'level': 2, 'parent': 2, 'edge': '是'},
        {'pos': (0.875, 0.55), 'text': '违约\n基尼系数: 0.15\n样本: 10', 'level': 2, 'parent': 2, 'edge': '否'},
    ]

    # 绘制边
    edges = [
        (0, 1, '是'), (0, 2, '否'),
        (1, 3, '是'), (1, 4, '否'),
        (2, 5, '是'), (2, 6, '否')
    ]

    for parent, child, label in edges:
        parent_pos = nodes[parent]['pos']
        child_pos = nodes[child]['pos']
        ax.annotate('', xy=child_pos, xytext=parent_pos,
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        mid_x = (parent_pos[0] + child_pos[0]) / 2
        mid_y = (parent_pos[1] + child_pos[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 绘制节点
    for node in nodes:
        if node['level'] == 0:
            color = '#3498db'  # 蓝色
            size = 8000
        elif node['level'] == 1:
            color = '#2ecc71'  # 绿色
            size = 6000
        else:
            if '违约' in node['text']:
                color = '#e74c3c'  # 红色
            else:
                color = '#95a5a6'  # 灰色
            size = 5000

        ax.scatter(node['pos'][0], node['pos'][1], s=size, c=color,
                  edgecolors='black', linewidth=2, zorder=5, alpha=0.8)
        ax.text(node['pos'][0], node['pos'][1], node['text'],
               fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title('决策树原理示意图', fontsize=18, pad=20)

    plt.tight_layout()
    plt.savefig('decision_tree_principle.png', dpi=300, bbox_inches='tight')
    print("\n决策树原理图已保存为 'decision_tree_principle.png'")

def visualize_feature_distributions(data, feature_columns):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        # 分别绘制正常和违约的数据
        normal_data = data[data['违约记录'] == 0][feature]
        default_data = data[data['违约记录'] == 1][feature]

        ax.hist(normal_data, bins=15, color='#2ecc71', alpha=0.6,
               label='正常', edgecolor='black')
        ax.hist(default_data, bins=15, color='#e74c3c', alpha=0.6,
               label='违约', edgecolor='black')

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{feature} 分布', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # 移除多余的子图
    if len(feature_columns) < len(axes):
        for idx in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('decision_tree_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\n特征分布图已保存为 'decision_tree_feature_distributions.png'")

def create_animation(X, y, feature_columns, max_depth=5):
    """创建决策树动画 - 逐步添加数据点展示树的生长过程"""
    print("\n" + "=" * 70)
    print("开始生成决策树动画...")
    print("=" * 70)

    # 创建保存帧的目录
    frames_dir = 'animation_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 只使用前两个特征进行2D可视化
    feature_x = feature_columns[0]  # 年龄
    feature_y = feature_columns[1]  # 收入

    X_2d = X[:, :2]

    # 打乱数据顺序，确保早期帧包含两个类别
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    # 交替排列两类样本
    indices = []
    min_len = min(len(class_0_indices), len(class_1_indices))
    for i in range(min_len):
        indices.append(class_0_indices[i])
        indices.append(class_1_indices[i])

    # 添加剩余样本
    remaining = class_0_indices[min_len:] if len(class_0_indices) > min_len else class_1_indices[min_len:]
    indices.extend(remaining)

    indices = np.array(indices)
    X_shuffled = X_2d[indices]
    y_shuffled = y[indices]

    print(f"总共有 {len(X_2d)} 个数据点")
    print(f"使用特征: {feature_x}, {feature_y}")
    print(f"最大深度: {max_depth}")

    # 从第6个点开始生成帧
    start_points = 6

    for n_points in range(start_points, len(X_shuffled) + 1):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 当前数据
        x_current = X_shuffled[:n_points]
        y_current = y_shuffled[:n_points]

        # 训练决策树
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            criterion='gini'
        )
        tree.fit(x_current, y_current)

        # 计算准确率
        y_pred = tree.predict(x_current)
        accuracy = accuracy_score(y_current, y_pred)

        # 获取树的信息
        n_nodes = tree.tree_.node_count
        n_leaves = tree.tree_.n_leaves
        tree_depth = tree.tree_.max_depth

        # 左图: 决策边界和分割区域
        # 创建网格
        x_min, x_max = x_current[:, 0].min() - 2, x_current[:, 0].max() + 2
        y_min, y_max = x_current[:, 1].min() - 2, x_current[:, 1].max() + 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                             np.arange(y_min, y_max, 0.5))

        # 预测网格点
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(['#FFB6C1', '#ADD8E6'])

        # 绘制决策区域
        ax1.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

        # 绘制待添加的点(灰色)
        if n_points < len(X_shuffled):
            ax1.scatter(X_shuffled[n_points:, 0], X_shuffled[n_points:, 1],
                       c='lightgray', s=40, alpha=0.4, edgecolors='gray',
                       label='待添加点', zorder=1)

        # 绘制当前已有的点
        class_0_mask = y_current[:-1] == 0
        class_1_mask = y_current[:-1] == 1

        ax1.scatter(x_current[:-1][class_0_mask, 0], x_current[:-1][class_0_mask, 1],
                   c='#e74c3c', s=80, alpha=0.8, edgecolors='black',
                   linewidth=1.5, label='正常客户', zorder=3)
        ax1.scatter(x_current[:-1][class_1_mask, 0], x_current[:-1][class_1_mask, 1],
                   c='#3498db', s=80, alpha=0.8, edgecolors='black',
                   linewidth=1.5, label='违约客户', zorder=3)

        # 最新添加的点高亮
        last_color = '#3498db' if y_current[-1] == 1 else '#e74c3c'
        ax1.scatter([x_current[-1, 0]], [x_current[-1, 1]], s=300, c=last_color,
                   alpha=0.95, edgecolors='yellow', linewidth=4,
                   zorder=5, label='新增点')

        # 绘制决策边界(分割线)
        def draw_tree_boundary(node, x_min, x_max, y_min, y_max):
            if tree.tree_.feature[node] != -2:  # 不是叶子节点
                feature_idx = tree.tree_.feature[node]
                threshold = tree.tree_.threshold[node]

                if feature_idx == 0:  # 年龄 (x轴)
                    ax1.axvline(x=threshold, ymin=0, ymax=1,
                              color='black', linestyle='--', linewidth=2, alpha=0.7)
                    draw_tree_boundary(tree.tree_.children_left[node],
                                     x_min, threshold, y_min, y_max)
                    draw_tree_boundary(tree.tree_.children_right[node],
                                     threshold, x_max, y_min, y_max)
                else:  # 收入 (y轴)
                    ax1.axhline(y=threshold, xmin=0, xmax=1,
                              color='black', linestyle='--', linewidth=2, alpha=0.7)
                    draw_tree_boundary(tree.tree_.children_left[node],
                                     x_min, x_max, y_min, threshold)
                    draw_tree_boundary(tree.tree_.children_right[node],
                                     x_min, x_max, threshold, y_max)

        # 绘制决策边界
        draw_tree_boundary(0, x_min, x_max, y_min, y_max)

        ax1.set_xlabel(f'{feature_x}', fontsize=13)
        ax1.set_ylabel(f'{feature_y}', fontsize=13)
        ax1.set_title(f'决策树分割边界 - {n_points} 个数据点', fontsize=14, fontweight='bold')
        ax1.legend(loc='center right', fontsize=9)
        ax1.grid(alpha=0.3)

        # 右图: 决策树结构可视化
        ax2.axis('off')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)

        # 绘制简化的决策树结构
        def plot_tree_node(node, x, y, width, level, max_level):
            if level > max_level or node >= n_nodes:
                return

            feature_idx = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]

            if feature_idx == -2:  # 叶子节点
                value = tree.tree_.value[node][0]
                pred_class = 0 if value[0] > value[1] else 1
                color = '#e74c3c' if pred_class == 0 else '#3498db'
                label = '正常' if pred_class == 0 else '违约'

                circle = plt.Circle((x, y), 0.4, color=color, alpha=0.7, ec='black', lw=2)
                ax2.add_patch(circle)
                ax2.text(x, y, f'{label}\n{int(value.sum())}个',
                        ha='center', va='center', fontsize=9, fontweight='bold')
            else:  # 决策节点
                feature_name = feature_x if feature_idx == 0 else feature_y

                circle = plt.Circle((x, y), 0.4, color='#2ecc71', alpha=0.7, ec='black', lw=2)
                ax2.add_patch(circle)
                ax2.text(x, y, f'{feature_name}\n≤{threshold:.1f}',
                        ha='center', va='center', fontsize=8, fontweight='bold')

                # 递归绘制子节点
                if width > 0.8:
                    left_child = tree.tree_.children_left[node]
                    right_child = tree.tree_.children_right[node]

                    # 左边连接线
                    ax2.plot([x, x - width/2], [y, y - 1.5], 'k-', lw=2)
                    ax2.text(x - width/4, y - 0.75, '是', ha='center', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
                    plot_tree_node(left_child, x - width/2, y - 1.5, width/2, level + 1, max_level)

                    # 右边连接线
                    ax2.plot([x, x + width/2], [y, y - 1.5], 'k-', lw=2)
                    ax2.text(x + width/4, y - 0.75, '否', ha='center', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
                    plot_tree_node(right_child, x + width/2, y - 1.5, width/2, level + 1, max_level)

        # 从根节点开始绘制
        plot_tree_node(0, 5, 9, 4, 0, min(4, max_depth))

        # 添加树的信息
        tree_info = f'决策树信息:\n'
        tree_info += f'数据点: {n_points}/{len(X_shuffled)}\n'
        tree_info += f'节点数: {n_nodes}\n'
        tree_info += f'叶节点数: {n_leaves}\n'
        tree_info += f'当前深度: {tree_depth}\n'
        tree_info += f'训练准确率: {accuracy*100:.1f}%\n'
        tree_info += f'正常客户: {(y_current==0).sum()}\n'
        tree_info += f'违约客户: {(y_current==1).sum()}'

        ax2.text(0.02, 0.98, tree_info, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title('决策树结构生长', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 保存帧
        frame_filename = os.path.join(frames_dir, f'frame_{n_points:03d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if n_points % 10 == 0 or n_points == len(X_shuffled):
            print(f"  已生成 {n_points}/{len(X_shuffled)} 帧 (节点: {n_nodes}, 深度: {tree_depth})")

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

        gif_path = 'decision_tree_animation.gif'
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=400,
                       loop=0)

        print(f"✅ GIF动画已保存为: {gif_path}")

    except ImportError:
        print("⚠️  PIL未安装，无法生成GIF")
        print("   安装方法: pip install Pillow")

    print("\n动画说明:")
    print("- 红色点: 正常客户 (Class 0)")
    print("- 蓝色点: 违约客户 (Class 1)")
    print("- 黄色光圈: 最新添加的数据点")
    print("- 灰色点: 待添加的数据点")
    print("- 左图: 决策边界演化")
    print("  * 彩色区域表示预测类别")
    print("  * 黑色虚线表示决策树的分割线")
    print("- 右图: 决策树结构生长")
    print("  * 绿色节点: 决策节点(进行判断)")
    print("  * 红色节点: 预测为正常")
    print("  * 蓝色节点: 预测为违约")
    print("- 观察决策树如何随着数据增加而生长")
    print("- 观察决策边界如何变得更加复杂")

def main():
    """主函数"""
    print("决策树算法示例 - 贷款违约预测")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'decision_tree_sample.csv'
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

    # 4. 分析不同深度的影响
    print("\n" + "=" * 70)
    print("步骤1: 分析树深度对性能的影响")
    print("=" * 70)
    best_depth = visualize_depth_impact(X_train, y_train, X_test, y_test)

    # 5. 训练最佳模型
    print("\n" + "=" * 70)
    print(f"步骤2: 训练决策树模型 (最佳深度={best_depth})")
    print("=" * 70)
    model = train_decision_tree(X_train, y_train, max_depth=best_depth)
    print("模型训练完成!")

    # 6. 评估模型
    print("\n" + "=" * 70)
    print("步骤3: 模型评估")
    print("=" * 70)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"准确率: {metrics['accuracy'] * 100:.2f}%")
    print(f"\n混淆矩阵:")
    print(metrics['conf_matrix'])
    print(f"\n分类报告:")
    print(metrics['report'])
    print(f"\nAUC值: {metrics['roc_auc']:.4f}")

    # 7. 打印决策树规则
    print_tree_rules(model, feature_columns)

    # 8. 特征重要性
    print("\n特征重要性:")
    for feature, importance in zip(feature_columns, model.feature_importances_):
        print(f"  {feature:20s}: {importance:.4f}")

    # 9. 示例预测
    print("\n" + "=" * 70)
    print("步骤4: 示例预测")
    print("=" * 70)

    sample_applicants = np.array([
        [30, 15, 5, 8, 0.2],   # 低风险申请人
        [35, 20, 8, 12, 0.5],  # 高风险申请人
        [28, 25, 6, 10, 0.35]  # 中等风险申请人
    ])

    sample_predictions = model.predict(sample_applicants)
    sample_probabilities = model.predict_proba(sample_applicants)

    descriptions = [
        "低风险 (收入稳定,负债率低)",
        "高风险 (负债率高)",
        "中等风险 (各项指标适中)"
    ]

    for i, (desc, pred, prob) in enumerate(zip(
        descriptions, sample_predictions, sample_probabilities), 1):
        result = "违约" if pred == 1 else "正常"
        print(f"\n申请人 {i} ({desc}):")
        print(f"  年龄: {sample_applicants[i-1][0]} 岁")
        print(f"  收入: {sample_applicants[i-1][1]} 万元")
        print(f"  工作年限: {sample_applicants[i-1][2]} 年")
        print(f"  信用卡额度: {sample_applicants[i-1][3]} 万元")
        print(f"  负债率: {sample_applicants[i-1][4]*100:.1f}%")
        print(f"  预测结果: {result}")
        print(f"  正常概率: {prob[0]*100:.2f}%")
        print(f"  违约概率: {prob[1]*100:.2f}%")

        # 获取决策路径
        decision_path = model.decision_path(sample_applicants[i-1:i+1])
        print(f"  决策路径: 经过 {decision_path.sum()} 个节点")

    # 10. 可视化
    print("\n" + "=" * 70)
    print("步骤5: 生成可视化图表")
    print("=" * 70)

    visualize_tree_structure(model, feature_columns, ['正常', '违约'])
    visualize_feature_importance(model, feature_columns)
    visualize_roc_curve(metrics)
    visualize_confusion_matrix(metrics['conf_matrix'])
    visualize_decision_boundary(model, X, y, feature_columns)
    visualize_tree_principle()
    visualize_feature_distributions(data, feature_columns)

    # 11. 生成动画
    print("\n" + "=" * 70)
    print("步骤6: 生成决策树动画")
    print("=" * 70)
    create_animation(X, y, feature_columns, max_depth=4)

    print("\n" + "=" * 70)
    print("决策树分析完成!")
    print("\n生成文件:")
    print("  - decision_tree_structure.png (决策树结构)")
    print("  - decision_tree_feature_importance.png (特征重要性)")
    print("  - decision_tree_roc_curve.png (ROC曲线)")
    print("  - decision_tree_confusion_matrix.png (混淆矩阵)")
    print("  - decision_tree_depth_analysis.png (深度分析)")
    print("  - decision_tree_boundary.png (决策边界)")
    print("  - decision_tree_principle.png (决策树原理)")
    print("  - decision_tree_feature_distributions.png (特征分布)")
    print("  - decision_tree_animation.gif (学习过程动画)")
    print("=" * 70)

if __name__ == "__main__":
    main()
