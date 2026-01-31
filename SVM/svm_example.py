"""
支持向量机(SVM)最小实例
使用鸢尾花数据作为示例: 根据花瓣和花萼尺寸分类品种
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
    print("\n品种分布:")
    print(data['品种'].value_counts())
    print(f"总样本数: {len(data)}")
    print(f"品种0 (山鸢尾): {sum(data['品种']==0)} 个")
    print(f"品种1 (维吉尼亚鸢尾): {sum(data['品种']==1)} 个")
    return data

def prepare_data(data):
    """准备训练数据"""
    feature_columns = ['花瓣长度(cm)', '花瓣宽度(cm)', '花萼长度(cm)', '花萼宽度(cm)']
    X = data[feature_columns].values
    y = data['品种'].values
    return X, y, feature_columns

def train_svm_models(X_train, y_train):
    """训练不同核函数的SVM模型"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}
    model_names = {
        'linear': '线性核 (Linear Kernel)',
        'rbf': 'RBF核 (RBF Kernel)',
        'poly': '多项式核 (Polynomial Kernel)',
        'sigmoid': 'Sigmoid核 (Sigmoid Kernel)'
    }

    # 训练不同核函数的SVM
    for kernel, name in model_names.items():
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=3, random_state=42, probability=True)
        else:
            model = SVC(kernel=kernel, random_state=42, probability=True)

        model.fit(X_train_scaled, y_train)
        models[kernel] = {
            'model': model,
            'name': name,
            'scaler': scaler
        }

    return models

def evaluate_model(model_info, X_test, y_test):
    """评估模型性能"""
    scaler = model_info['scaler']
    model = model_info['model']

    # 标准化测试数据
    X_test_scaled = scaler.transform(X_test)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['山鸢尾', '维吉尼亚鸢尾'])

    # ROC曲线
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
        'roc_auc': roc_auc,
        'n_support': model.n_support_
    }

def visualize_comparison(models, X_test, y_test):
    """可视化不同核函数的对比"""
    # 评估所有模型
    results = {}
    for kernel, model_info in models.items():
        results[kernel] = evaluate_model(model_info, X_test, y_test)

    # 创建对比图
    fig = plt.figure(figsize=(16, 10))

    # 子图1: 准确率对比
    ax1 = plt.subplot(2, 3, 1)
    kernels = list(results.keys())
    accuracies = [results[k]['accuracy'] for k in kernels]
    aucs = [results[k]['roc_auc'] for k in kernels]

    x = np.arange(len(kernels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='准确率',
                    color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, aucs, width, label='AUC值',
                    color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('分数', fontsize=12)
    ax1.set_title('不同核函数性能对比', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['线性核', 'RBF核', '多项式核', 'Sigmoid核'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.7, 1.05])

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # 子图2-5: 各核函数的ROC曲线
    for idx, (kernel, result) in enumerate(results.items(), 2):
        ax = plt.subplot(2, 3, idx)

        ax.plot(result['fpr'], result['tpr'],
               label=f'{models[kernel]["name"]}\n(AUC = {result["roc_auc"]:.4f})',
               linewidth=2.5, color='#3498db')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='随机分类器')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假正率', fontsize=11)
        ax.set_ylabel('真正率', fontsize=11)
        ax.set_title(models[kernel]['name'], fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    # 子图6: 支持向量数量对比
    ax6 = plt.subplot(2, 3, 6)
    support_counts = [sum(models[k]['model'].n_support_) for k in kernels]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = ax6.bar(kernels, support_counts, color=colors, edgecolor='black')

    ax6.set_ylabel('支持向量数量', fontsize=12)
    ax6.set_title('各核函数的支持向量数量', fontsize=14)
    ax6.set_xticklabels(['线性核', 'RBF核', '多项式核', 'Sigmoid核'])
    ax6.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('svm_comparison.png', dpi=300, bbox_inches='tight')
    print("\n模型对比图已保存为 'svm_comparison.png'")

def visualize_decision_boundary(model_info, X, y, feature_columns):
    """可视化决策边界(使用前两个特征)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 只使用前两个特征进行2D可视化
    X_2d = X[:, :2]

    # 标准化
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # 训练2D模型
    model_2d = SVC(kernel='rbf', random_state=42)
    model_2d.fit(X_2d_scaled, y)

    # 创建网格
    x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
    y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 不同核函数的决策边界
    kernels_config = [
        ('linear', '线性核决策边界', {'kernel': 'linear'}),
        ('rbf', 'RBF核决策边界', {'kernel': 'rbf', 'gamma': 'scale'}),
        ('poly', '多项式核决策边界', {'kernel': 'poly', 'degree': 3}),
        ('sigmoid', 'Sigmoid核决策边界', {'kernel': 'sigmoid'})
    ]

    cmap_light = ListedColormap(['#FFB6C1', '#ADD8E6'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    for idx, (kernel, title, params) in enumerate(kernels_config):
        ax = axes[idx]

        # 训练模型
        clf = SVC(random_state=42, probability=True, **params)
        clf.fit(X_2d_scaled, y)

        # 预测网格点
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

        # 绘制数据点
        ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], c=y, cmap=cmap_bold,
                  edgecolors='black', s=50, alpha=0.8)

        # 标记支持向量
        ax.scatter(X_2d_scaled[clf.support_, 0], X_2d_scaled[clf.support_, 1],
                  c='none', edgecolors='yellow', s=150, linewidths=2,
                  label=f'支持向量 ({len(clf.support_)}个)')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(feature_columns[0] + ' (标准化)', fontsize=11)
        ax.set_ylabel(feature_columns[1] + ' (标准化)', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("决策边界图已保存为 'svm_decision_boundary.png'")

def visualize_feature_distributions(data, feature_columns):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        # 分别绘制两个品种的数据
        class_0 = data[data['品种'] == 0][feature]
        class_1 = data[data['品种'] == 1][feature]

        ax.hist(class_0, bins=15, color='#e74c3c', alpha=0.6,
               label='山鸢尾', edgecolor='black')
        ax.hist(class_1, bins=15, color='#3498db', alpha=0.6,
               label='维吉尼亚鸢尾', edgecolor='black')

        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(f'{feature} 分布', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('svm_feature_distribution.png', dpi=300, bbox_inches='tight')
    print("特征分布图已保存为 'svm_feature_distribution.png'")

def visualize_svm_concept():
    """绘制SVM原理图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 最大间隔原理
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 模拟数据点
    class_a = np.array([[2, 3], [3, 2], [2, 5], [3, 4], [4, 3]])
    class_b = np.array([[7, 8], [8, 7], [7, 6], [8, 8], [6, 7]])

    ax1.scatter(class_a[:, 0], class_a[:, 1], s=200, c='red',
               edgecolors='black', label='类别A', zorder=3)
    ax1.scatter(class_b[:, 0], class_b[:, 1], s=200, c='blue',
               edgecolors='black', label='类别B', zorder=3)

    # 决策边界
    x = np.linspace(0, 10, 100)
    y = -x + 10
    ax1.plot(x, y, 'k-', linewidth=3, label='决策边界', zorder=2)

    # 间隔边界
    y1 = -x + 11
    y2 = -x + 9
    ax1.plot(x, y1, 'g--', linewidth=2, label='间隔边界', zorder=1)
    ax1.plot(x, y2, 'g--', linewidth=2, zorder=1)

    # 支持向量
    support_vectors_a = np.array([[3, 4], [4, 3]])
    support_vectors_b = np.array([[7, 6], [7, 8]])
    ax1.scatter(support_vectors_a[:, 0], support_vectors_a[:, 1], s=300,
               facecolors='none', edgecolors='yellow', linewidths=3,
               label='支持向量', zorder=4)
    ax1.scatter(support_vectors_b[:, 0], support_vectors_b[:, 1], s=300,
               facecolors='none', edgecolors='yellow', linewidths=3, zorder=4)

    ax1.set_xlabel('特征1', fontsize=13)
    ax1.set_ylabel('特征2', fontsize=13)
    ax1.set_title('SVM最大间隔原理', fontsize=15)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(alpha=0.3)

    # 右图: 核函数示意图
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-1, 4)

    x = np.linspace(-5, 5, 200)

    # RBF核
    y_rbf = np.exp(-x**2 / 2)
    ax2.plot(x, y_rbf, 'b-', linewidth=3, label='RBF核函数')

    # 标记关键点
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.scatter([0], [1], s=200, c='orange', edgecolors='black',
               zorder=5, label='中心点')

    ax2.set_xlabel('距离', fontsize=13)
    ax2.set_ylabel('相似度', fontsize=13)
    ax2.set_title('RBF核函数示意图', fontsize=15)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('svm_concept.png', dpi=300, bbox_inches='tight')
    print("SVM原理图已保存为 'svm_concept.png'")

def main():
    """主函数"""
    print("支持向量机(SVM)示例 - 鸢尾花品种分类")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'svm_sample.csv'
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

    # 4. 训练不同核函数的SVM模型
    print("\n开始训练SVM模型...")
    models = train_svm_models(X_train, y_train)
    print("模型训练完成!")

    # 5. 评估所有模型
    print(f"\n{'='*70}")
    print("各核函数模型性能对比:")
    print(f"{'='*70}")

    results = {}
    for kernel, model_info in models.items():
        result = evaluate_model(model_info, X_test, y_test)
        results[kernel] = result
        print(f"\n{model_info['name']}:")
        print(f"  准确率: {result['accuracy']*100:.2f}%")
        print(f"  AUC值: {result['roc_auc']:.4f}")
        print(f"  支持向量数: {sum(result['n_support'])}")

    # 6. 选择最佳模型(RBF核通常效果最好)
    best_model = models['rbf']
    best_result = results['rbf']

    print(f"\n{'='*70}")
    print("最佳模型详情 (RBF核):")
    print(f"{'='*70}")
    print(f"准确率: {best_result['accuracy']*100:.2f}%")
    print(f"\n混淆矩阵:")
    print(best_result['conf_matrix'])
    print(f"\n分类报告:")
    print(best_result['report'])

    # 7. 示例预测
    print(f"\n{'='*70}")
    print("示例预测:")
    print(f"{'='*70}")

    sample_flowers = np.array([
        [1.5, 0.3, 5.0, 3.5],   # 山鸢尾特征
        [4.7, 1.4, 6.5, 3.0],   # 维吉尼亚鸢尾特征
        [3.0, 1.0, 5.5, 3.0]    # 中间特征
    ])

    scaler = best_model['scaler']
    sample_scaled = scaler.transform(sample_flowers)
    sample_predictions = best_model['model'].predict(sample_scaled)
    sample_probabilities = best_model['model'].predict_proba(sample_scaled)

    variety_names = ['山鸢尾 (Class 0)', '维吉尼亚鸢尾 (Class 1)',
                    '不确定样本 (边界)']

    for i, (desc, pred, prob) in enumerate(zip(variety_names,
                                               sample_predictions,
                                               sample_probabilities), 1):
        variety = "山鸢尾" if pred == 0 else "维吉尼亚鸢尾"
        print(f"\n样本{i} ({desc}):")
        print(f"  花瓣长度: {sample_flowers[i-1][0]} cm")
        print(f"  花瓣宽度: {sample_flowers[i-1][1]} cm")
        print(f"  花萼长度: {sample_flowers[i-1][2]} cm")
        print(f"  花萼宽度: {sample_flowers[i-1][3]} cm")
        print(f"  预测品种: {variety}")
        print(f"  山鸢尾概率: {prob[0]*100:.2f}%")
        print(f"  维吉尼亚鸢尾概率: {prob[1]*100:.2f}%")

    # 8. 可视化
    print(f"\n{'='*70}")
    print("生成可视化图表...")
    print(f"{'='*70}")
    visualize_comparison(models, X_test, y_test)
    visualize_decision_boundary(best_model, X, y, feature_columns)
    visualize_feature_distributions(data, feature_columns)
    visualize_svm_concept()

    print(f"\n{'='*70}")
    print("SVM分析完成!")
    print("\n生成文件:")
    print("  - svm_comparison.png (核函数性能对比)")
    print("  - svm_decision_boundary.png (决策边界可视化)")
    print("  - svm_feature_distribution.png (特征分布)")
    print("  - svm_concept.png (SVM原理图)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
