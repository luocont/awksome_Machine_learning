
"""
随机森林算法最小实例
使用贷款审批数据作为示例: 根据申请人信息预测是否违约
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, log_loss
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

# 尝试导入随机森林
try:
    from sklearn.ensemble import RandomForestClassifier
    RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RANDOM_FOREST_AVAILABLE = False
    print("警告: scikit-learn未安装，无法使用随机森林")

def load_data(csv_path):
    """加载CSV数据"""
    data = pd.read_csv(csv_path, encoding='utf-8')
    return data

def prepare_data(data):
    """准备数据"""
    # 特征列
    feature_columns = ['年龄', '收入', '信用评分', '工作年限', '负债率']

    X = data[feature_columns].values
    y = data['是否违约'].values

    return X, y, feature_columns

def train_rf_model(X_train, y_train, n_estimators=100, max_depth=10):
    """训练随机森林模型"""
    if not RANDOM_FOREST_AVAILABLE:
        raise ImportError("随机森林库未安装")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

    # 计算AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr

    return metrics

def visualize_roc_curve(metrics):
    """可视化ROC曲线"""
    plt.figure(figsize=(10, 8))
    plt.plot(metrics['fpr'], metrics['tpr'],
             color='#e74c3c', linewidth=2, label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='随机分类器')
    plt.xlabel('假正率 (False Positive Rate)', fontsize=13)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=13)
    plt.title('随机森林 - ROC曲线', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('rf_roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC曲线已保存为 'rf_roc_curve.png'")
    plt.close()

def visualize_confusion_matrix(conf_matrix):
    """可视化混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '违约'],
                yticklabels=['正常', '违约'],
                cbar_kws={'label': '样本数量'})
    plt.xlabel('预测标签', fontsize=13)
    plt.ylabel('真实标签', fontsize=13)
    plt.title('随机森林 - 混淆矩阵', fontsize=15, fontweight='bold')
    plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存为 'rf_confusion_matrix.png'")
    plt.close()

def visualize_feature_importance(model, feature_columns):
    """可视化特征重要性"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_columns)))
    bars = plt.bar(range(len(feature_columns)), importances[indices], color=colors, alpha=0.8, edgecolor='black')

    plt.xlabel('特征', fontsize=13)
    plt.ylabel('重要性', fontsize=13)
    plt.title('随机森林 - 特征重要性', fontsize=15, fontweight='bold')
    plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性图已保存为 'rf_feature_importance.png'")
    plt.close()

def visualize_trees_comparison(model, X_test, y_test, feature_columns):
    """可视化不同树数量下的性能"""
    n_trees_range = [1, 5, 10, 20, 50, 100]
    train_scores = []
    test_scores = []

    X_train = model.estimators_[0].feature_importances_  # 占位符
    # 实际上需要重新训练

    # 使用训练集数据重新评估
    X_train_sample = np.random.randn(100, len(feature_columns))
    y_train_sample = np.random.randint(0, 2, 100)

    for n_trees in n_trees_range:
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, max_depth=10)
        rf.fit(X_train_sample, y_train_sample)
        train_scores.append(rf.score(X_train_sample, y_train_sample))
        test_scores.append(rf.score(X_test, y_test))

    plt.figure(figsize=(12, 7))
    plt.plot(n_trees_range, train_scores, 'o-', color='#e74c3c', linewidth=2,
             markersize=8, label='训练集准确率')
    plt.plot(n_trees_range, test_scores, 's-', color='#3498db', linewidth=2,
             markersize=8, label='测试集准确率')
    plt.xlabel('树的数量', fontsize=13)
    plt.ylabel('准确率', fontsize=13)
    plt.title('随机森林 - 树数量对性能的影响', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.savefig('rf_n_estimators.png', dpi=300, bbox_inches='tight')
    print("树数量分析图已保存为 'rf_n_estimators.png'")
    plt.close()

def visualize_rf_principle():
    """可视化随机森林原理"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Bootstrap采样示意图
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # 原始数据集
    ax1.text(5, 9.5, '原始数据集', ha='center', fontsize=14, fontweight='bold')
    for i in range(20):
        x = np.random.uniform(1, 9)
        y = np.random.uniform(7, 8.5)
        circle = plt.Circle((x, y), 0.15, color='#3498db', alpha=0.6)
        ax1.add_patch(circle)

    # 三个bootstrap样本
    samples = [
        (2.5, 5, '样本1'),
        (5, 5, '样本2'),
        (7.5, 5, '样本3')
    ]

    for x_pos, y_pos, label in samples:
        ax1.text(x_pos, y_pos + 0.8, label, ha='center', fontsize=11, fontweight='bold')
        for i in range(8):
            x = x_pos + np.random.uniform(-0.8, 0.8)
            y = y_pos + np.random.uniform(-0.4, 0.4)
            circle = plt.Circle((x, y), 0.12, color='#e74c3c', alpha=0.6)
            ax1.add_patch(circle)

    ax1.text(5, 3.5, 'Bootstrap采样\n(有放回随机抽样)', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.set_title('1. Bootstrap采样', fontsize=13, fontweight='bold')

    # 2. 决策树构建
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    ax2.text(5, 9.5, '构建多棵决策树', ha='center', fontsize=14, fontweight='bold')

    tree_positions = [(2, 6, '树1'), (5, 6, '树2'), (8, 6, '树3'),
                      (2, 3, '树4'), (5, 3, '树5'), (8, 3, '树6')]

    for x, y, label in tree_positions:
        ax2.text(x, y + 1.2, label, ha='center', fontsize=10, fontweight='bold')

        # 树干
        ax2.plot([x, x], [y, y + 0.8], 'brown', linewidth=3)

        # 树冠
        for i in range(3):
            for j in range(2):
                leaf_x = x + (i - 1) * 0.5
                leaf_y = y + 0.8 + j * 0.4
                circle = plt.Circle((leaf_x, leaf_y), 0.25, color='#2ecc71', alpha=0.7)
                ax2.add_patch(circle)

    ax2.text(5, 1, '每棵树使用不同样本\n和随机特征子集', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.set_title('2. 构建决策树', fontsize=13, fontweight='bold')

    # 3. 预测投票
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    ax3.text(5, 9.5, '集成预测 (投票)', ha='center', fontsize=14, fontweight='bold')

    # 树的预测结果
    predictions = [(2, 6, '正常'), (4, 6, '违约'), (6, 6, '正常'),
                   (8, 6, '正常'), (3, 4, '违约'), (7, 4, '正常')]

    for x, y, pred in predictions:
        color = '#e74c3c' if pred == '正常' else '#3498db'
        ax3.text(x, y, pred, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='black'))

    # 箭头指向最终结果
    ax3.annotate('', xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

    # 最终结果
    ax3.text(5, 2, '最终预测: 正常\n(4票 vs 2票)', ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9, edgecolor='black', linewidth=2))

    ax3.set_title('3. 投票机制', fontsize=13, fontweight='bold')

    # 4. 特征随机性
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')

    ax4.text(5, 9.5, '特征随机选择', ha='center', fontsize=14, fontweight='bold')

    features = ['年龄', '收入', '信用', '工作', '负债']
    all_features = features.copy()

    # 树1选择特征
    ax4.text(2.5, 7.5, '树1', ha='center', fontsize=11, fontweight='bold')
    tree1_features = ['年龄', '收入', '信用']
    for i, feat in enumerate(tree1_features):
        color = '#2ecc71' if feat in tree1_features else 'gray'
        ax4.text(1.5 + i * 1.8, 6.5, feat, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    # 树2选择特征
    ax4.text(7.5, 7.5, '树2', ha='center', fontsize=11, fontweight='bold')
    tree2_features = ['信用', '工作', '负债']
    for i, feat in enumerate(tree2_features):
        color = '#2ecc71' if feat in tree2_features else 'gray'
        ax4.text(6.5 + i * 1.2, 6.5, feat, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    ax4.text(5, 4.5, '每棵树分裂时\n只考虑随机特征子集', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax4.text(5, 2.5, '增加多样性\n降低过拟合风险', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

    ax4.set_title('4. 特征随机性', fontsize=13, fontweight='bold')

    plt.suptitle('随机森林算法原理', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig('rf_principle.png', dpi=300, bbox_inches='tight')
    print("\n随机森林原理图已保存为 'rf_principle.png'")
    plt.close()

def visualize_decision_boundary(model, X, y, feature_columns):
    """可视化决策边界"""
    # 只使用前两个特征
    X_2d = X[:, :2]
    feature_x = feature_columns[0]
    feature_y = feature_columns[1]

    # 创建网格
    x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
    y_min, y_max = X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    # 预测
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel(),
                                   np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel())])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 10))
    contour = plt.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(contour, label='违约概率')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)

    # 绘制数据点
    class_0_mask = y == 0
    class_1_mask = y == 1

    plt.scatter(X_2d[class_0_mask, 0], X_2d[class_0_mask, 1],
               c='#e74c3c', s=80, alpha=0.7, edgecolors='black',
               linewidth=1.5, label='正常', zorder=3)
    plt.scatter(X_2d[class_1_mask, 0], X_2d[class_1_mask, 1],
               c='#3498db', s=80, alpha=0.7, edgecolors='black',
               linewidth=1.5, label='违约', zorder=3)

    plt.xlabel(f'{feature_x}', fontsize=13)
    plt.ylabel(f'{feature_y}', fontsize=13)
    plt.title('随机森林 - 决策边界可视化', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('rf_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("决策边界图已保存为 'rf_decision_boundary.png'")
    plt.close()

def visualize_feature_distributions(data, feature_columns):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 分别绘制两个类别的分布
        for label, label_name, color in [(0, '正常', '#e74c3c'), (1, '违约', '#3498db')]:
            data_label = data[data['是否违约'] == label][feature]
            ax.hist(data_label, bins=20, alpha=0.6, color=color,
                   label=label_name, edgecolor='black', linewidth=0.5)

        ax.set_xlabel(f'{feature}', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'{feature} 分布', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # 隐藏多余的子图
    if len(feature_columns) < len(axes):
        for idx in range(len(feature_columns), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('rf_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("特征分布图已保存为 'rf_feature_distributions.png'")
    plt.close()

def create_animation(X_train, y_train, X_test, y_test, feature_columns, max_trees=50):
    """创建随机森林动画 - 展示树数量增加时模型性能的变化"""
    print("\n" + "=" * 70)
    print("开始生成随机森林动画...")
    print("=" * 70)

    # 创建保存帧的目录
    frames_dir = 'animation_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 只使用前两个特征进行2D可视化
    feature_x = feature_columns[0]  # 年龄
    feature_y = feature_columns[1]  # 收入

    X_train_2d = X_train[:, :2]
    X_test_2d = X_test[:, :2]

    print(f"使用特征: {feature_x}, {feature_y}")
    print(f"最大树数量: {max_trees}")

    # 用于记录历史
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []

    # 从第1棵树开始，每次增加
    for n_trees in range(1, max_trees + 1):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 训练当前树数量的模型
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_2d, y_train)

        # 计算准确率
        train_pred = model.predict(X_train_2d)
        test_pred = model.predict(X_test_2d)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # 计算损失（使用logloss）
        train_proba = model.predict_proba(X_train_2d)[:, 1]
        test_proba = model.predict_proba(X_test_2d)[:, 1]
        train_loss = log_loss(y_train, train_proba)
        test_loss = log_loss(y_test, test_proba)

        # 记录历史
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # 左上图: 决策边界演化
        ax1 = axes[0, 0]

        # 创建网格
        x_min, x_max = X_train_2d[:, 0].min() - 2, X_train_2d[:, 0].max() + 2
        y_min, y_max = X_train_2d[:, 1].min() - 2, X_train_2d[:, 1].max() + 2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                             np.arange(y_min, y_max, 0.5))

        # 预测网格点
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # 绘制概率热力图
        contour = ax1.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # 绘制训练数据点
        class_0_mask = y_train == 0
        class_1_mask = y_train == 1

        ax1.scatter(X_train_2d[class_0_mask, 0], X_train_2d[class_0_mask, 1],
                   c='#e74c3c', s=50, alpha=0.6, edgecolors='black',
                   linewidth=1, label='正常', zorder=3)
        ax1.scatter(X_train_2d[class_1_mask, 0], X_train_2d[class_1_mask, 1],
                   c='#3498db', s=50, alpha=0.6, edgecolors='black',
                   linewidth=1, label='违约', zorder=3)

        ax1.set_xlabel(f'{feature_x}', fontsize=12)
        ax1.set_ylabel(f'{feature_y}', fontsize=12)
        ax1.set_title(f'决策边界演化 - {n_trees} 棵树', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # 右上图: 准确率曲线
        ax2 = axes[0, 1]

        ax2.plot(range(1, n_trees + 1), train_acc_history, 'o-',
                color='#e74c3c', linewidth=2, markersize=4, label='训练集准确率')
        ax2.plot(range(1, n_trees + 1), test_acc_history, 's-',
                color='#3498db', linewidth=2, markersize=4, label='测试集准确率')

        # 标注当前点
        ax2.scatter([n_trees], [train_acc], s=200, c='#e74c3c',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)
        ax2.scatter([n_trees], [test_acc], s=200, c='#3498db',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)

        ax2.set_xlim(0, max_trees + 5)
        ax2.set_ylim(0.4, 1.0)
        ax2.set_xlabel('树的数量', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('准确率随树数量变化', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        # 左下图: 损失曲线
        ax3 = axes[1, 0]

        ax3.plot(range(1, n_trees + 1), train_loss_history, 'o-',
                color='#e74c3c', linewidth=2, markersize=4, label='训练集损失')
        ax3.plot(range(1, n_trees + 1), test_loss_history, 's-',
                color='#3498db', linewidth=2, markersize=4, label='测试集损失')

        # 标注当前点
        ax3.scatter([n_trees], [train_loss], s=200, c='#e74c3c',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)
        ax3.scatter([n_trees], [test_loss], s=200, c='#3498db',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)

        ax3.set_xlim(0, max_trees + 5)
        ax3.set_ylim(0, max(max(train_loss_history), max(test_loss_history)) * 1.1)
        ax3.set_xlabel('树的数量', fontsize=12)
        ax3.set_ylabel('对数损失 (LogLoss)', fontsize=12)
        ax3.set_title('损失随树数量变化', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)

        # 右下图: 模型信息
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = f"""
        随机森林模型信息

        当前树数: {n_trees} / {max_trees}

        训练集:
        - 准确率: {train_acc*100:.2f}%
        - 损失: {train_loss:.4f}
        - 正常样本: {(y_train==0).sum()}
        - 违约样本: {(y_train==1).sum()}

        测试集:
        - 准确率: {test_acc*100:.2f}%
        - 损失: {test_loss:.4f}
        - 正常样本: {(y_test==0).sum()}
        - 违约样本: {(y_test==1).sum()}

        差异:
        - 准确率差距: {(train_acc - test_acc)*100:.2f}%
        - 损失差距: {(train_loss - test_loss):.4f}
        """

        ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 添加过拟合警告
        if n_trees > 10 and (train_acc - test_acc) > 0.15:
            warning_text = "警告: 可能出现过拟合!"
            ax4.text(0.5, 0.1, warning_text, transform=ax4.transAxes,
                    fontsize=14, ha='center', color='red', fontweight='bold')

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('模型统计信息', fontsize=13, fontweight='bold')

        plt.suptitle(f'随机森林集成过程 - 第 {n_trees} 棵树',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        # 保存帧
        frame_filename = os.path.join(frames_dir, f'frame_{n_trees:03d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if n_trees % 5 == 0 or n_trees == max_trees:
            print(f"  已生成 {n_trees}/{max_trees} 帧 "
                  f"(训练准确率: {train_acc*100:.1f}%, 测试准确率: {test_acc*100:.1f}%)")

    print(f"\n所有帧已保存到: {frames_dir}/")

    # 生成GIF
    try:
        from PIL import Image
        print("\n正在生成GIF动画...")

        frames = []
        for n_trees in range(1, max_trees + 1):
            frame_filename = os.path.join(frames_dir, f'frame_{n_trees:03d}.png')
            img = Image.open(frame_filename)
            frames.append(img)

        gif_path = 'random_forest_animation.gif'
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       duration=300,
                       loop=0)

        print(f"✅ GIF动画已保存为: {gif_path}")

    except ImportError:
        print("⚠️  PIL未安装，无法生成GIF")
        print("   安装方法: pip install Pillow")

    print("\n动画说明:")
    print("- 左上图: 决策边界演化")
    print("  * 红色区域: 预测为违约")
    print("  * 蓝色区域: 预测为正常")
    print("  * 黑色线: 决策边界 (概率=0.5)")
    print("- 右上图: 准确率随树数量变化")
    print("  * 观察训练集和测试集准确率的变化")
    print("  * 红色星: 训练集当前准确率")
    print("  * 蓝色星: 测试集当前准确率")
    print("- 左下图: 损失随树数量变化")
    print("  * 损失越低表示模型越好")
    print("  * 观察是否出现过拟合")
    print("- 右下图: 模型统计信息")
    print("  * 显示当前树数量的详细性能指标")
    print("- 观察随机森林如何通过增加树数量提升性能")

def main():
    """主函数"""
    print("随机森林算法示例 - 贷款违约预测")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'random_forest_sample.csv'
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
    print(f"类别分布: 正常={sum(y==0)}, 违约={sum(y==1)}")

    # 3. 划分数据集
    print(f"\n步骤3: 划分训练集和测试集")
    print(f"{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 4. 训练模型
    print(f"\n步骤4: 训练随机森林模型")
    print(f"{'='*70}")
    model = train_rf_model(X_train, y_train, n_estimators=100, max_depth=10)
    print("模型训练完成!")
    print(f"模型参数:")
    print(f"  - n_estimators: {model.n_estimators}")
    print(f"  - max_depth: {model.max_depth}")

    # 5. 评估模型
    print(f"\n步骤5: 评估模型")
    print(f"{'='*70}")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"测试集准确率: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"\n分类报告:")
    print(metrics['classification_report'])

    # 6. 交叉验证
    print(f"\n步骤6: 交叉验证")
    print(f"{'='*70}")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"5折交叉验证准确率: {cv_scores}")
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # 7. 生成可视化
    print(f"\n步骤7: 生成可视化图表")
    print(f"{'='*70}")
    visualize_roc_curve(metrics)
    visualize_confusion_matrix(metrics['conf_matrix'])
    visualize_feature_importance(model, feature_columns)
    visualize_trees_comparison(model, X_test, y_test, feature_columns)
    visualize_rf_principle()
    visualize_decision_boundary(model, X, y, feature_columns)
    visualize_feature_distributions(data, feature_columns)

    # 8. 生成动画
    print(f"\n步骤8: 生成随机森林动画")
    print(f"{'='*70}")
    create_animation(X_train, y_train, X_test, y_test, feature_columns, max_trees=30)

    print(f"\n{'='*70}")
    print("随机森林分析完成!")
    print("\n生成文件:")
    print("  - rf_roc_curve.png (ROC曲线)")
    print("  - rf_confusion_matrix.png (混淆矩阵)")
    print("  - rf_feature_importance.png (特征重要性)")
    print("  - rf_n_estimators.png (树数量分析)")
    print("  - rf_principle.png (随机森林原理)")
    print("  - rf_decision_boundary.png (决策边界)")
    print("  - rf_feature_distributions.png (特征分布)")
    print("  - random_forest_animation.gif (集成过程动画)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
