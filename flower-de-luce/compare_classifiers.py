import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, KBinsDiscretizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from matplotlib.colors import ListedColormap

# 加载数据
iris = load_iris()
X = iris.data[:, 2:]  # 只使用后两个特征（花瓣长度和宽度）
y = iris.target
feature_names = ['Petal Length', 'Petal Width']

# 创建网格
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
)

# 选择4个有代表性的分类器
classifiers = {
    "Logistic regression\n(C=0.01)": LogisticRegression(C=0.1, max_iter=1000),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=0.5, n_components=50, random_state=1),
        LogisticRegression(C=10, max_iter=1000),
    ),
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10, max_iter=1000),
    ),
}

# 创建图形 - 现在有5列（颜色条列 + 3个类别概率图 + Max class图）
n_classifiers = len(classifiers)
fig, axes = plt.subplots(n_classifiers, 5, figsize=(18, 4.5 * n_classifiers))

# 定义颜色方案
# 决策边界颜色
decision_colors = ['#FFDDDD', '#DDFFDD', '#DDDDFF']  # 浅红, 浅绿, 浅蓝
cmap_light = ListedColormap(decision_colors)

# 数据点颜色
data_colors = ['red', 'green', 'blue']  # 红, 绿, 蓝
cmap_data = ListedColormap(data_colors)

# 为概率图定义颜色映射
prob_cmap = plt.cm.Blues

# 为每个分类器生成可视化
for row_idx, (clf_name, clf) in enumerate(classifiers.items()):
    print(f"处理分类器: {clf_name}")
    
    # 训练模型
    clf.fit(X, y)
    
    # 预测整个网格
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 获取预测类别和概率
    Z = clf.predict(grid_points).reshape(xx.shape)
    probs = clf.predict_proba(grid_points)
    
    # ========== 列0: 颜色条 ==========
    ax = axes[row_idx, 0]
    # 隐藏坐标轴
    ax.set_axis_off()
    
    # 创建概率颜色条
    # 添加概率颜色条的标题
    ax.text(0.5, 0.85, 'Probability', ha='center', va='center', 
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    # 创建一个模拟的图像用于概率颜色条
    prob_norm = plt.Normalize(0, 1)
    prob_sm = plt.cm.ScalarMappable(cmap=prob_cmap, norm=prob_norm)
    prob_sm.set_array([])
    
    # 添加概率颜色条
    prob_cbar = plt.colorbar(prob_sm, ax=ax, orientation='vertical', 
                            fraction=0.35, pad=0.02, aspect=15, location='left')
    prob_cbar.set_label('', fontsize=9)
    
    # 创建类别颜色条
    # 添加类别颜色条的标题
    ax.text(0.5, 0.35, 'Classes', ha='center', va='center', 
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    # 创建一个模拟的图像用于类别颜色条
    class_norm = plt.Normalize(0, 2)
    class_sm = plt.cm.ScalarMappable(cmap=cmap_data, norm=class_norm)
    class_sm.set_array([])
    
    # 添加类别颜色条
    class_cbar = plt.colorbar(class_sm, ax=ax, orientation='vertical', 
                             fraction=0.35, pad=0.02, aspect=15, location='right')
    class_cbar.set_ticks([0, 1, 2])
    class_cbar.set_ticklabels(['Class 0', 'Class 1', 'Class 2'])
    class_cbar.set_label('', fontsize=9)
    
    # 设置颜色条列的整体标题（只在第一行）
    if row_idx == 0:
        ax.set_title('Color Bars', fontsize=12, fontweight='bold', pad=20)
    
    # ========== 列1-3: 每个类别的概率 ==========
    for class_idx in range(3):
        ax = axes[row_idx, class_idx + 1]  # 注意：现在从列1开始
        
        # 获取当前类别的概率
        prob_class = probs[:, class_idx].reshape(xx.shape)
        
        # 使用蓝色系颜色映射
        im_prob = ax.contourf(xx, yy, prob_class, levels=20, cmap=prob_cmap, alpha=0.8)
    
        # 绘制属于该类别的数据点，用对应颜色
        class_mask = (y == class_idx)
        point_color = data_colors[class_idx]
        ax.scatter(X[class_mask, 0], X[class_mask, 1], 
                  c=point_color, edgecolors='k', s=60, marker='o', 
                  label=f'Class {class_idx}')
    
        ax.set_title(f'Class {class_idx}', fontsize=11)
        if row_idx == n_classifiers - 1:
            ax.set_xlabel(feature_names[0])
        if class_idx == 0:
            ax.set_ylabel(feature_names[1])
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # ========== 列4: 最大类别 ==========
    ax = axes[row_idx, 4]
    
    # 找到每个点的最大概率类别
    max_class = np.argmax(probs, axis=1).reshape(xx.shape)
    
    # 绘制最大类别区域
    ax.contourf(xx, yy, max_class, cmap=cmap_light, alpha=0.8)
    
    # 计算置信度（最大概率值）
    max_prob = np.max(probs, axis=1).reshape(xx.shape)
    
    # 绘制低置信度区域（概率<0.7）
    low_confidence = max_prob < 0.7
    ax.contourf(xx, yy, low_confidence, colors='none', 
                hatches=['...'], alpha=0.3)
    
    # 绘制数据点
    scatter_max = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_data, 
                            edgecolors='k', s=60, alpha=0.9)
    
    ax.set_title(f'{clf_name}\nMax class (confidence <0.7 hatched)', fontsize=11, fontweight='bold')
    if row_idx == n_classifiers - 1:
        ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.grid(True, alpha=0.3)

plt.suptitle('Classifier Comparison: Class Probabilities and Max Class Predictions', 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print("\n可视化完成！")