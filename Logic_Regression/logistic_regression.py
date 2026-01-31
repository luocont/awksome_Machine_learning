"""
逻辑回归 - 多特征随机数据版本
使用多个学习指标预测学生考试是否通过
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    print("\n通过情况统计:")
    print(data['通过状态'].value_counts())
    print(f"通过率: {data['通过状态'].mean() * 100:.2f}%")
    print(f"总样本数: {len(data)}")
    return data

def prepare_data(data):
    """准备训练数据"""
    feature_columns = ['学习时间(小时)', '出勤率(%)', '作业完成率(%)', '考前复习(小时)']
    X = data[feature_columns].values
    y = data['通过状态'].values
    return X, y, feature_columns

def train_model(X_train, y_train):
    """训练逻辑回归模型"""
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """评估模型性能"""
    # 标准化测试数据
    X_test_scaled = scaler.transform(X_test)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['不通过', '通过'])

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
        'roc_auc': roc_auc
    }

def visualize_results(metrics):
    """可视化结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 混淆矩阵
    ax1 = axes[0]
    conf_matrix = metrics['conf_matrix']
    im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)

    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax1.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=18)

    ax1.set_ylabel('真实标签', fontsize=13)
    ax1.set_xlabel('预测标签', fontsize=13)
    ax1.set_title(f'混淆矩阵\n(准确率: {metrics["accuracy"]*100:.2f}%)', fontsize=14)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['不通过', '通过'])
    ax1.set_yticklabels(['不通过', '通过'])

    # 右图: ROC曲线
    ax2 = axes[1]
    ax2.plot(metrics['fpr'], metrics['tpr'], color='blue',
             linewidth=2.5, label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--',
             linewidth=2, label='随机分类器')

    # 添加最佳点标注
    optimal_idx = np.argmax(metrics['tpr'] - metrics['fpr'])
    optimal_threshold = metrics['fpr'][optimal_idx]
    ax2.plot(metrics['fpr'][optimal_idx], metrics['tpr'][optimal_idx], 'go',
             markersize=10, label='最佳阈值点')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('假正率 (False Positive Rate)', fontsize=12)
    ax2.set_ylabel('真正率 (True Positive Rate)', fontsize=12)
    ax2.set_title('ROC 曲线', fontsize=14)
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logistic_regression_multi_result.png', dpi=300, bbox_inches='tight')
    print("\n可视化结果已保存为 'logistic_regression_multi_result.png'")

def visualize_feature_importance(model, feature_columns):
    """可视化特征重要性"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 获取系数绝对值
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)

    # 绘制水平条形图
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.barh(range(len(importance)), importance[indices], color=colors)

    # 添加数值标签
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{importance[idx]:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('特征重要性 (系数绝对值)', fontsize=13)
    ax.set_title('逻辑回归 - 特征重要性分析', fontsize=15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性图已保存为 'feature_importance.png'")

def visualize_data_distribution(data, feature_columns):
    """可视化数据分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors_pass = '#2ecc71'
    colors_fail = '#e74c3c'

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        # 分别绘制通过和未通过的数据
        pass_data = data[data['通过状态'] == 1][feature]
        fail_data = data[data['通过状态'] == 0][feature]

        ax.hist(fail_data, bins=15, color=colors_fail, alpha=0.6,
               label='不通过', edgecolor='black')
        ax.hist(pass_data, bins=15, color=colors_pass, alpha=0.6,
               label='通过', edgecolor='black')

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('样本数', fontsize=11)
        ax.set_title(f'{feature} 分布', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("数据分布图已保存为 'data_distribution.png'")

def main():
    """主函数"""
    print("逻辑回归示例 - 多特征学生考试通过预测")
    print("=" * 70)

    # 1. 加载数据
    csv_path = 'logistic_regression_sample.csv'
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

    # 4. 训练模型
    print("\n开始训练模型...")
    model, scaler = train_model(X_train, y_train)
    print("模型训练完成!")

    # 5. 模型参数
    print(f"\n{'='*70}")
    print("模型参数分析:")
    print(f"{'='*70}")
    for feature, coef in zip(feature_columns, model.coef_[0]):
        print(f"{feature:20s} : 系数 = {coef:8.4f}")
    print(f"{'截距':20s} : {model.intercept_[0]:8.4f}")

    # 6. 评估模型
    print(f"\n{'='*70}")
    print("模型评估:")
    print(f"{'='*70}")
    metrics = evaluate_model(model, scaler, X_test, y_test)
    print(f"准确率: {metrics['accuracy'] * 100:.2f}%")
    print(f"\n混淆矩阵:")
    print(metrics['conf_matrix'])
    print(f"\n分类报告:")
    print(metrics['report'])
    print(f"\nAUC值: {metrics['roc_auc']:.4f}")

    # 7. 示例预测
    print(f"\n{'='*70}")
    print("示例预测:")
    print(f"{'='*70}")

    sample_students = np.array([
        [3.0, 70, 55, 1.0],   # 学习时间短
        [6.0, 85, 78, 2.5],   # 中等水平
        [9.0, 98, 98, 4.0]    # 学习时间长
    ])

    # 标准化
    sample_scaled = scaler.transform(sample_students)
    sample_predictions = model.predict(sample_scaled)
    sample_probabilities = model.predict_proba(sample_scaled)

    descriptions = [
        "学习时间短, 出勤率低",
        "中等学习水平",
        "学习认真, 准备充分"
    ]

    for i, (desc, pred, prob) in enumerate(zip(descriptions,
                                               sample_predictions,
                                               sample_probabilities), 1):
        status = "✓ 通过" if pred == 1 else "✗ 不通过"
        print(f"\n学生{i} ({desc}):")
        print(f"  学习时间: {sample_students[i-1][0]}小时")
        print(f"  出勤率: {sample_students[i-1][1]}%")
        print(f"  作业完成率: {sample_students[i-1][2]}%")
        print(f"  考前复习: {sample_students[i-1][3]}小时")
        print(f"  预测结果: {status}")
        print(f"  通过概率: {prob[1]*100:.2f}%")

    # 8. 可视化
    print(f"\n{'='*70}")
    print("生成可视化图表...")
    print(f"{'='*70}")
    visualize_results(metrics)
    visualize_feature_importance(model, feature_columns)
    visualize_data_distribution(data, feature_columns)

    print(f"\n{'='*70}")
    print("逻辑回归分析完成!")
    print("\n生成文件:")
    print("  - logistic_regression_multi_result.png (混淆矩阵+ROC曲线)")
    print("  - feature_importance.png (特征重要性)")
    print("  - data_distribution.png (数据分布)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
