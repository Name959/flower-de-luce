# binary_3d_decision_surface.py
"""
二分类3D决策边界可视化
使用Iris数据集的前两个类别，在后三个特征空间中展示决策边界和淡色半透明的分割面
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def create_binary_3d_decision_surface():
    """
    创建二分类3D决策边界可视化
    前两个分类，后三个特征，一个三维图
    """
    # 1. 加载和准备数据
    print("加载Iris数据集...")
    iris = load_iris()
    X = iris.data  # 所有4个特征
    y = iris.target
    
    # 选择前两个类别（类别0和1）
    binary_mask = (y == 0) | (y == 1)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    # 选择后三个特征（特征索引1, 2, 3）
    # 特征1: Sepal Width, 特征2: Petal Length, 特征3: Petal Width
    feature_indices = [1, 2, 3]
    X_3d = X_binary[:, feature_indices]
    feature_names = ['Sepal Width', 'Petal Length', 'Petal Width']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)
    
    # 2. 训练SVM分类器
    print("训练SVM分类器...")
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_scaled, y_binary)
    
    # 3. 创建3D网格用于绘制分割面
    print("创建3D网格用于分割面...")
    # 确定每个特征的范围
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5
    
    # 创建分割面的网格
    grid_resolution = 30
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    
    # 4. 创建可视化
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义颜色
    class0_color = '#FF6B6B'     # 类别0: 浅红色
    class1_color = '#4ECDC4'     # 类别1: 青色
    
    # 5. 绘制数据点
    print("绘制数据点...")
    # 类别0的数据点
    mask_class0 = (y_binary == 0)
    ax.scatter(X_scaled[mask_class0, 0], 
               X_scaled[mask_class0, 1], 
               X_scaled[mask_class0, 2],
               c=class0_color,
               edgecolors='k',
               s=100,
               alpha=0.9,
               depthshade=True,
               label='Class 0 (Setosa)')
    
    # 类别1的数据点
    mask_class1 = (y_binary == 1)
    ax.scatter(X_scaled[mask_class1, 0], 
               X_scaled[mask_class1, 1], 
               X_scaled[mask_class1, 2],
               c=class1_color,
               edgecolors='k',
               s=100,
               alpha=0.9,
               depthshade=True,
               label='Class 1 (Versicolor)')
    
    # 6. 绘制淡色半透明的分割面
    print("绘制淡色半透明的分割面...")
    
    # 获取线性SVM的参数
    w = clf.coef_[0]  # 权重向量 [w0, w1, w2]
    b = clf.intercept_[0]  # 偏置项
    
    # 计算分割面方程: w0*x + w1*y + w2*z + b = 0
    # 解出 z = -(w0*x + w1*y + b) / w2
    # 需要确保 w2 不为 0
    if abs(w[2]) > 1e-10:
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]
        
        # 绘制分割面 - 淡色半透明
        surf = ax.plot_surface(xx, yy, zz,
                             color='#E0E0E0',  # 淡灰色
                             alpha=0.4,  # 40%透明度
                             rstride=1, 
                             cstride=1,
                             linewidth=0.5,
                             edgecolor='gray',
                             antialiased=True)
        
        # 为了让分割面更明显，可以添加一个轮廓
        ax.contour(xx, yy, zz, 
                  levels=[0],  # 决策边界
                  colors='gray',
                  linestyles='-',
                  linewidths=1,
                  alpha=0.5)
        
        print(f"分割面方程: {w[0]:.3f}*x + {w[1]:.3f}*y + {w[2]:.3f}*z + {b:.3f} = 0")
    else:
        # 如果w2接近0，分割面几乎是垂直的
        print("注意: w2接近0，分割面几乎垂直")
        # 绘制一个近似的平面
        zz = np.zeros_like(xx)
        surf = ax.plot_surface(xx, yy, zz,
                             color='#E0E0E0',
                             alpha=0.4,
                             rstride=1, 
                             cstride=1,
                             linewidth=0.5,
                             edgecolor='gray',
                             antialiased=True)
    
    # 7. 设置图形属性
    ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
    ax.set_zlabel(feature_names[2], fontsize=12, fontweight='bold')
    
    ax.set_title('Binary Classification: Decision Boundary in 3D Feature Space\n' +
                 'Linear SVM with Transparent Decision Surface', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 设置视角 - 调整视角以更好地显示分割面
    ax.view_init(elev=25, azim=135)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 设置背景平面样式
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    # 8. 添加颜色条显示决策函数的置信度
    # 计算所有数据点的决策函数值
    decision_values = clf.decision_function(X_scaled)
    
    # 创建颜色映射
    from matplotlib.cm import coolwarm
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=decision_values.min(), vmax=decision_values.max())
    sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=norm)
    sm.set_array([])
    
    # 添加颜色条
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Decision Function Value', fontsize=10)
    cbar.ax.text(0.5, -0.15, 'Class 0', transform=cbar.ax.transAxes, 
                ha='center', va='top', fontsize=9, color=class0_color)
    cbar.ax.text(0.5, 1.15, 'Class 1', transform=cbar.ax.transAxes, 
                ha='center', va='bottom', fontsize=9, color=class1_color)
    
    # 调整布局
    plt.tight_layout()
    
    # 9. 显示模型信息
    print("\n" + "="*60)
    print("模型信息:")
    print("="*60)
    print(f"类别: Class 0 (Setosa) vs Class 1 (Versicolor)")
    print(f"样本数: {len(y_binary)} (Class 0: {sum(y_binary==0)}, Class 1: {sum(y_binary==1)})")
    print(f"特征: {feature_names}")
    print(f"分类器: Linear SVM")
    print(f"权重向量 w: [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}]")
    print(f"偏置项 b: {b:.4f}")
    
    # 计算准确率
    train_accuracy = clf.score(X_scaled, y_binary)
    print(f"训练准确率: {train_accuracy:.2%}")
    print("="*60)
    
    plt.show()
    
    return {
        'X_scaled': X_scaled,
        'y_binary': y_binary,
        'feature_names': feature_names,
        'classifier': clf,
        'figure': fig,
        'axes': ax
    }

# 创建更简单的可视化版本（如果主函数有问题）
def create_simple_3d_plot():
    """
    创建一个简单的3D图，确保分割面清晰可见
    """
    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 选择前两个类别和后三个特征
    binary_mask = (y == 0) | (y == 1)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    X_3d = X_binary[:, [1, 2, 3]]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)
    
    # 训练线性SVM
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_scaled, y_binary)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    mask_class0 = (y_binary == 0)
    mask_class1 = (y_binary == 1)
    
    ax.scatter(X_scaled[mask_class0, 0], X_scaled[mask_class0, 1], X_scaled[mask_class0, 2],
               c='red', edgecolors='k', s=80, alpha=0.9, label='Class 0 (Setosa)')
    ax.scatter(X_scaled[mask_class1, 0], X_scaled[mask_class1, 1], X_scaled[mask_class1, 2],
               c='blue', edgecolors='k', s=80, alpha=0.9, label='Class 1 (Versicolor)')
    
    # 绘制分割面
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    # 创建网格
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25),
                         np.linspace(y_min, y_max, 25))
    
    if abs(w[2]) > 1e-10:
        zz = -(w[0] * xx + w[1] * yy + b) / w[2]
        
        # 绘制淡色半透明的分割面
        ax.plot_surface(xx, yy, zz,
                       color='lightgray',
                       alpha=0.4,  # 40%透明度
                       rstride=1, cstride=1,
                       linewidth=0.3,
                       edgecolor='gray')
    
    # 设置图形属性
    ax.set_xlabel('Sepal Width')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Petal Width')
    ax.set_title('Binary Classification with Transparent Decision Surface')
    ax.legend()
    ax.view_init(elev=25, azim=135)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("二分类3D决策边界可视化")
    print("="*60)
    print("前两个分类，后三个特征")
    print("一个三维图包含：")
    print("  - 两个分类的数据点（不同颜色）")
    print("  - 淡色半透明的分割面")
    print("="*60)
    
    try:
        results = create_binary_3d_decision_surface()
        print("\n可视化完成!")
    except Exception as e:
        print(f"主方法出错: {e}")
        print("尝试使用简化版本...")
        create_simple_3d_plot()