"""
XGBoost算法最小实例
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
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装! 请运行: pip install xgboost")
    print("将使用sklearn的GradientBoostingClassifier作为替代...")
    from sklearn.ensemble import GradientBoostingClassifier

def load_data(csv_path):
    """加载CSV数据"""
    data = pd.read_csv(csv_path, encoding='utf-8')
    print("=" * 70)
    print("数据预览:")
    print(data.head(10))
    print("\n数据统计信息:")
    print(data.describe())
    print("\n违约情况统计:")
    print(data['违约'].value_counts())
    print(f"总样本数: {len(data)}")
    print(f"正常客户 (0): {sum(data['违约']==0)} 个")
    print(f"违约客户 (1): {sum(data['违约']==1)} 个")
    print(f"违约率: {data['违约'].mean() * 100:.2f}%")
    return data

def prepare_data(data):
    """准备数据"""
    feature_columns = ['年龄', '年收入(万元)', '工作年限', '教育程度', '信用卡额度(万元)',
                      '负债率', '月消费(万元)', '信用分', '银行账户数', '贷款记录']
    X = data[feature_columns].values
    y = data['违约'].values
    return X, y, feature_columns

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, **params):
    """训练XGBoost模型"""
    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            **params
        )
    else:
        # 使用sklearn的GradientBoosting作为替代
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42,
            **params
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

def find_optimal_n_estimators(X_train, y_train, X_test, y_test):
    """寻找最优的树的数量"""
    print("\n正在测试不同的树的数量...")

    n_estimators_range = range(10, 210, 20)
    train_scores = []
    test_scores = []

    for n_est in n_estimators_range:
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

        print(f"n_estimators={n_est:3d} | 训练: {train_score*100:5.2f}% | 测试: {test_score*100:5.2f}%")

    best_n = n_estimators_range[np.argmax(test_scores)]

    return n_estimators_range, train_scores, test_scores, best_n

def find_optimal_depth(X_train, y_train, X_test, y_test):
    """寻找最优的树深度"""
    print("\n正在测试不同的树深度...")

    depth_range = range(3, 11)
    train_scores = []
    test_scores = []
    cv_scores = []

    for depth in depth_range:
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=depth,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=depth,
                random_state=42
            )

        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        train_scores.append(train_score)
        test_scores.append(test_score)
        cv_scores.append(cv_score)

        print(f"max_depth={depth:2d} | 训练: {train_score*100:5.2f}% | "
             f"测试: {test_score*100:5.2f}% | 交叉验证: {cv_score*100:5.2f}%")

    best_depth = depth_range[np.argmax(test_scores)]

    return depth_range, train_scores, test_scores, cv_scores, best_depth

def visualize_n_estimators(n_estimators_range, train_scores, test_scores, best_n):
    """可视化树数量的影响"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(n_estimators_range, train_scores, 'o-', color='#e74c3c',
           linewidth=2, markersize=6, label='训练集准确率')
    ax.plot(n_estimators_range, test_scores, 's-', color='#3498db',
           linewidth=2, markersize=6, label='测试集准确率')

    best_idx = list(n_estimators_range).index(best_n)
    best_score = test_scores[best_idx]

    ax.axvline(x=best_n, color='orange', linestyle='--',
              linewidth=2, alpha=0.7, label=f'最佳数量: {best_n}')
    ax.scatter([best_n], [best_score], s=300, c='orange',
              marker='*', edgecolors='black', zorder=5,
              label=f'最佳测试准确率: {best_score*100:.2f}%')

    ax.set_xlabel('树的数量 (n_estimators)', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('XGBoost: 树的数量对性能的影响', fontsize=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('xgboost_n_estimators.png', dpi=300, bbox_inches='tight')
    print("\n树数量分析图已保存为 'xgboost_n_estimators.png'")

def visualize_depth_analysis(depth_range, train_scores, test_scores, cv_scores, best_depth):
    """可视化深度分析"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(depth_range, train_scores, 'o-', color='#e74c3c',
           linewidth=2, markersize=6, label='训练集准确率')
    ax.plot(depth_range, test_scores, 's-', color='#3498db',
           linewidth=2, markersize=6, label='测试集准确率')
    ax.plot(depth_range, cv_scores, '^-', color='#2ecc71',
           linewidth=2, markersize=6, label='交叉验证准确率')

    best_idx = list(depth_range).index(best_depth)
    best_score = test_scores[best_idx]

    ax.axvline(x=best_depth, color='orange', linestyle='--',
              linewidth=2, alpha=0.7, label=f'最佳深度: {best_depth}')
    ax.scatter([best_depth], [best_score], s=300, c='orange',
              marker='*', edgecolors='black', zorder=5,
              label=f'最佳测试准确率: {best_score*100:.2f}%')

    ax.set_xlabel('树的最大深度 (max_depth)', fontsize=13)
    ax.set_ylabel('准确率', fontsize=13)
    ax.set_title('XGBoost: 深度对性能的影响', fontsize=15)
    ax.legend(fontsize=11, loc='center right')
    ax.grid(alpha=0.3)
    ax.set_xticks(depth_range)

    plt.tight_layout()
    plt.savefig('xgboost_depth_analysis.png', dpi=300, bbox_inches='tight')
    print("\n深度分析图已保存为 'xgboost_depth_analysis.png'")

def visualize_feature_importance(model, feature_columns):
    """可视化特征重要性"""
    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))

    indices = np.argsort(importance)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))

    bars = ax.barh(range(len(importance)), importance[indices],
                   color=colors[indices], edgecolor='black', alpha=0.8)

    for i, (bar, idx) in enumerate(zip(bars, indices)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{importance[idx]:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_columns[i] for i in indices])
    ax.set_xlabel('特征重要性', fontsize=13)
    ax.set_title('XGBoost - 特征重要性分析', fontsize=15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n特征重要性图已保存为 'xgboost_feature_importance.png'")

def visualize_roc_curve(metrics):
    """可视化ROC曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics['fpr'], metrics['tpr'],
           color='#3498db', linewidth=2.5,
           label=f'ROC曲线 (AUC = {metrics["roc_auc"]:.4f})')

    ax.plot([0, 1], [0, 1], color='red', linestyle='--',
           linewidth=2, label='随机分类器')

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
    plt.savefig('xgboost_roc_curve.png', dpi=300, bbox_inches='tight')
    print("\nROC曲线图已保存为 'xgboost_roc_curve.png'")

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
    plt.savefig('xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n混淆矩阵图已保存为 'xgboost_confusion_matrix.png'")

def visualize_learning_curve(model, X_train, y_train, X_test, y_test):
    """可视化学习曲线"""
    if not XGBOOST_AVAILABLE:
        print("\n学习曲线图需要XGBoost库支持,跳过...")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train,
             eval_set=eval_set,
             verbose=False)

    # 获取评估结果
    results = model.evals_result()
    train_logloss = results['validation_0']['logloss']
    test_logloss = results['validation_1']['logloss']

    epochs = len(train_logloss) + 1
    x_axis = range(1, epochs)

    ax.plot(x_axis, train_logloss, 'o-', color='#e74c3c',
           linewidth=2, markersize=4, label='训练集LogLoss')
    ax.plot(x_axis, test_logloss, 's-', color='#3498db',
           linewidth=2, markersize=4, label='测试集LogLoss')

    ax.set_xlabel('迭代次数', fontsize=13)
    ax.set_ylabel('对数损失 (LogLoss)', fontsize=13)
    ax.set_title('XGBoost学习曲线', fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('xgboost_learning_curve.png', dpi=300, bbox_inches='tight')
    print("\n学习曲线图已保存为 'xgboost_learning_curve.png'")

def visualize_boosting_process():
    """绘制Boosting原理图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 生成模拟数据
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = np.sin(X) + 0.3 * np.random.randn(100)

    # Boosting迭代过程
    residuals = y_true.copy()
    predictions = np.zeros_like(y_true)

    iterations = [1, 3, 5, 10, 20, 50]

    for idx, n_iter in enumerate(iterations):
        ax = axes[idx // 3, idx % 3]

        # 模拟boosting过程
        for _ in range(n_iter):
            # 简单的决策树桩(阶梯函数)
            threshold = np.random.choice(X)
            stump = np.where(X > threshold, 1, -1) * 0.1

            # 更新预测和残差
            predictions += stump
            residuals = y_true - predictions

        # 绘制
        ax.scatter(X, y_true, c='gray', alpha=0.3, s=20, label='真实数据')
        ax.plot(X, predictions, 'r-', linewidth=2, label='Boosting预测')
        ax.plot(X, np.sin(X), 'b--', linewidth=1.5, alpha=0.5, label='真实函数')

        ax.set_title(f'迭代 {n_iter} 次', fontsize=12)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_ylim(-2, 2)

    plt.suptitle('XGBoost/Boosting 迭代过程演示', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('xgboost_boosting_process.png', dpi=300, bbox_inches='tight')
    print("\nBoosting过程图已保存为 'xgboost_boosting_process.png'")

def visualize_feature_distributions(data, feature_columns):
    """可视化特征分布"""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]

        normal_data = data[data['违约'] == 0][feature]
        default_data = data[data['违约'] == 1][feature]

        ax.hist(normal_data, bins=12, color='#2ecc71', alpha=0.6,
               label='正常', edgecolor='black')
        ax.hist(default_data, bins=12, color='#e74c3c', alpha=0.6,
               label='违约', edgecolor='black')

        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('频数', fontsize=9)
        ax.set_title(f'{feature[:8]}...', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('xgboost_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\n特征分布图已保存为 'xgboost_feature_distributions.png'")

def visualize_xgboost_principle():
    """绘制XGBoost原理图"""
    fig = plt.figure(figsize=(16, 10))

    # 创建网格布局
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 子图1: Boosting原理
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    principle_text = """
    XGBoost 原理

    1. 初始化: 预测值为均值或0

    2. 迭代训练:
       - 计算残差(真实值 - 当前预测值)
       - 训练新树预测残差
       - 更新预测值 = 旧预测 + 学习率 × 新树预测

    3. 正则化:
       - 控制树复杂度(深度、叶子节点数)
       - L1/L2正则化
       - 防止过拟合

    4. 输出: 所有树的预测之和
    """

    ax1.text(0.1, 0.5, principle_text, fontsize=12, va='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图2: 目标函数
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    objective_text = r"""
    XGBoost 目标函数

    Obj = Loss + Regularization

    其中:
    - Loss: 损失函数(如LogLoss)
    - Regularization: 正则化项

    正则化项:
    Ω(f) = γT + 0.5λ||w||²

    T: 树的叶子节点数
    w: 叶子节点权重
    γ, λ: 正则化参数
    """

    ax2.text(0.1, 0.5, objective_text, fontsize=11, va='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 子图3: 特征重要性
    ax3 = fig.add_subplot(gs[1, :])

    features = ['信用分', '负债率', '收入', '工作年限', '教育程度', '年龄']
    importance = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax3.barh(features, importance, color=colors, edgecolor='black')

    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
               f'{imp:.0%}', ha='left', va='center', fontsize=10)

    ax3.set_xlabel('重要性', fontsize=12)
    ax3.set_title('特征重要性示例', fontsize=13)
    ax3.grid(axis='x', alpha=0.3)

    # 子图4: 模型对比
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    comparison_text = """
    XGBoost vs 其他算法

    算法              准确率    训练速度    预测速度    可解释性
    ──────────────────────────────────────────────────────
    决策树             中        快          快          高
    随机森林           高        慢          中          低
    XGBoost           很高       中          快          中
    神经网络          很高       慢          快          低

    XGBoost 优势:
    ✓ 极高的准确率
    ✓ 内置正则化防止过拟合
    ✓ 支持并行计算
    ✓ 处理缺失值
    ✓ 可解释性强(特征重要性)
    """

    ax4.text(0.1, 0.5, comparison_text, fontsize=11, va='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle('XGBoost 算法原理', fontsize=18, y=0.98)

    plt.savefig('xgboost_principle.png', dpi=300, bbox_inches='tight')
    print("\nXGBoost原理图已保存为 'xgboost_principle.png'")

def create_animation(X_train, y_train, X_test, y_test, feature_columns, max_iterations=50):
    """创建XGBoost动画 - 展示Boosting迭代过程中模型性能的变化"""
    print("\n" + "=" * 70)
    print("开始生成XGBoost动画...")
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
    print(f"最大迭代次数: {max_iterations}")

    # 用于记录历史
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []

    # 从第1次迭代开始，每次增加
    for iteration in range(1, max_iterations + 1):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 训练当前迭代次数的模型
        if XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=iteration,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=iteration,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
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
        ax1.set_title(f'决策边界演化 - 迭代 {iteration} 次', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # 右上图: 准确率曲线
        ax2 = axes[0, 1]

        ax2.plot(range(1, iteration + 1), train_acc_history, 'o-',
                color='#e74c3c', linewidth=2, markersize=4, label='训练集准确率')
        ax2.plot(range(1, iteration + 1), test_acc_history, 's-',
                color='#3498db', linewidth=2, markersize=4, label='测试集准确率')

        # 标注当前点
        ax2.scatter([iteration], [train_acc], s=200, c='#e74c3c',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)
        ax2.scatter([iteration], [test_acc], s=200, c='#3498db',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)

        ax2.set_xlim(0, max_iterations + 5)
        ax2.set_ylim(0.4, 1.0)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('准确率随迭代变化', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        # 左下图: 损失曲线
        ax3 = axes[1, 0]

        ax3.plot(range(1, iteration + 1), train_loss_history, 'o-',
                color='#e74c3c', linewidth=2, markersize=4, label='训练集损失')
        ax3.plot(range(1, iteration + 1), test_loss_history, 's-',
                color='#3498db', linewidth=2, markersize=4, label='测试集损失')

        # 标注当前点
        ax3.scatter([iteration], [train_loss], s=200, c='#e74c3c',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)
        ax3.scatter([iteration], [test_loss], s=200, c='#3498db',
                   marker='*', edgecolors='black', linewidth=2, zorder=5)

        ax3.set_xlim(0, max_iterations + 5)
        ax3.set_ylim(0, max(max(train_loss_history), max(test_loss_history)) * 1.1)
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('对数损失 (LogLoss)', fontsize=12)
        ax3.set_title('损失随迭代变化', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)

        # 右下图: 模型信息
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = f"""
        XGBoost 模型信息

        当前迭代: {iteration} / {max_iterations}

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
        if iteration > 10 and (train_acc - test_acc) > 0.15:
            warning_text = "警告: 可能出现过拟合!"
            ax4.text(0.5, 0.1, warning_text, transform=ax4.transAxes,
                    fontsize=14, ha='center', color='red', fontweight='bold')

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('模型统计信息', fontsize=13, fontweight='bold')

        plt.suptitle(f'XGBoost Boosting 过程 - 第 {iteration} 次迭代',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()

        # 保存帧
        frame_filename = os.path.join(frames_dir, f'frame_{iteration:03d}.png')
        plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if iteration % 5 == 0 or iteration == max_iterations:
            print(f"  已生成 {iteration}/{max_iterations} 帧 "
                  f"(训练准确率: {train_acc*100:.1f}%, 测试准确率: {test_acc*100:.1f}%)")

    print(f"\n所有帧已保存到: {frames_dir}/")

    # 生成GIF
    try:
        from PIL import Image
        print("\n正在生成GIF动画...")

        frames = []
        for iteration in range(1, max_iterations + 1):
            frame_filename = os.path.join(frames_dir, f'frame_{iteration:03d}.png')
            img = Image.open(frame_filename)
            frames.append(img)

        gif_path = 'xgboost_animation.gif'
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
    print("- 右上图: 准确率随迭代变化")
    print("  * 观察训练集和测试集准确率的变化")
    print("  * 红色星: 训练集当前准确率")
    print("  * 蓝色星: 测试集当前准确率")
    print("- 左下图: 损失随迭代变化")
    print("  * 损失越低表示模型越好")
    print("  * 观察是否出现过拟合")
    print("- 右下图: 模型统计信息")
    print("  * 显示当前迭代的详细性能指标")
    print("- 观察XGBoost如何通过迭代逐步提升性能")

def main():
    """主函数"""
    print("XGBoost算法示例 - 贷款违约预测")
    print("=" * 70)

    if not XGBOOST_AVAILABLE:
        print("\n注意: XGBoost库未安装,使用sklearn的GradientBoostingClassifier替代")
        print("安装XGBoost以获得更好的性能: pip install xgboost\n")
    else:
        print(f"\nXGBoost版本: {xgb.__version__}")

    # 1. 加载数据
    csv_path = 'xgboost_sample.csv'
    data = load_data(csv_path)

    # 2. 准备数据
    X, y, feature_columns = prepare_data(data)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n数据集划分:")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")

    # 4. 分析树数量的影响
    print("\n" + "=" * 70)
    print("步骤1: 分析树数量对性能的影响")
    print("=" * 70)
    n_est_range, train_scores_n, test_scores_n, best_n = \
        find_optimal_n_estimators(X_train, y_train, X_test, y_test)
    visualize_n_estimators(n_est_range, train_scores_n, test_scores_n, best_n)

    # 5. 分析树深度的影响
    print("\n" + "=" * 70)
    print("步骤2: 分析树深度对性能的影响")
    print("=" * 70)
    depth_range, train_scores_d, test_scores_d, cv_scores_d, best_depth = \
        find_optimal_depth(X_train, y_train, X_test, y_test)
    visualize_depth_analysis(depth_range, train_scores_d, test_scores_d,
                           cv_scores_d, best_depth)

    # 6. 训练最佳模型
    print("\n" + "=" * 70)
    print(f"步骤3: 训练XGBoost模型")
    print(f"  - 最佳树数量: {best_n}")
    print(f"  - 最佳深度: {best_depth}")
    print("=" * 70)

    model = train_xgboost(X_train, y_train,
                         n_estimators=best_n,
                         max_depth=best_depth)
    print("模型训练完成!")

    # 7. 评估模型
    print("\n" + "=" * 70)
    print("步骤4: 模型评估")
    print("=" * 70)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"准确率: {metrics['accuracy'] * 100:.2f}%")
    print(f"\n混淆矩阵:")
    print(metrics['conf_matrix'])
    print(f"\n分类报告:")
    print(metrics['report'])
    print(f"\nAUC值: {metrics['roc_auc']:.4f}")

    # 8. 特征重要性
    print("\n特征重要性:")
    for feature, importance in sorted(zip(feature_columns, model.feature_importances_),
                                    key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {importance:.4f}")

    # 9. 示例预测
    print("\n" + "=" * 70)
    print("步骤5: 示例预测")
    print("=" * 70)

    sample_applicants = np.array([
        [30, 18, 5, 2, 10, 0.20, 1.2, 715, 2, 1],   # 低风险
        [35, 20, 8, 2, 12, 0.50, 1.5, 665, 3, 3],   # 高风险
        [28, 25, 6, 3, 15, 0.35, 1.4, 690, 2, 2]    # 中等风险
    ])

    sample_predictions = model.predict(sample_applicants)
    sample_probabilities = model.predict_proba(sample_applicants)

    descriptions = [
        "低风险 (信用良好,负债率低)",
        "高风险 (负债率高,有多次贷款记录)",
        "中等风险 (各项指标适中)"
    ]

    for i, (desc, pred, prob) in enumerate(zip(
        descriptions, sample_predictions, sample_probabilities), 1):
        result = "违约" if pred == 1 else "正常"
        print(f"\n申请人 {i} ({desc}):")
        print(f"  年龄: {sample_applicants[i-1][0]} 岁")
        print(f"  收入: {sample_applicants[i-1][1]} 万元")
        print(f"  工作年限: {sample_applicants[i-1][2]} 年")
        print(f"  负债率: {sample_applicants[i-1][5]*100:.1f}%")
        print(f"  信用分: {sample_applicants[i-1][7]}")
        print(f"  预测结果: {result}")
        print(f"  正常概率: {prob[0]*100:.2f}%")
        print(f"  违约概率: {prob[1]*100:.2f}%")

    # 10. 可视化
    print("\n" + "=" * 70)
    print("步骤6: 生成可视化图表")
    print("=" * 70)

    visualize_roc_curve(metrics)
    visualize_confusion_matrix(metrics['conf_matrix'])
    visualize_feature_importance(model, feature_columns)
    visualize_learning_curve(model, X_train, y_train, X_test, y_test)
    visualize_boosting_process()
    visualize_feature_distributions(data, feature_columns)
    visualize_xgboost_principle()

    # 11. 生成动画
    print("\n" + "=" * 70)
    print("步骤7: 生成XGBoost动画")
    print("=" * 70)
    create_animation(X_train, y_train, X_test, y_test, feature_columns, max_iterations=30)

    print("\n" + "=" * 70)
    print("XGBoost分析完成!")
    print("\n生成文件:")
    print("  - xgboost_n_estimators.png (树数量分析)")
    print("  - xgboost_depth_analysis.png (深度分析)")
    print("  - xgboost_feature_importance.png (特征重要性)")
    print("  - xgboost_roc_curve.png (ROC曲线)")
    print("  - xgboost_confusion_matrix.png (混淆矩阵)")
    print("  - xgboost_learning_curve.png (学习曲线)")
    print("  - xgboost_boosting_process.png (Boosting过程)")
    print("  - xgboost_feature_distributions.png (特征分布)")
    print("  - xgboost_principle.png (XGBoost原理)")
    print("  - xgboost_animation.gif (Boosting迭代过程动画)")
    print("=" * 70)

if __name__ == "__main__":
    main()
