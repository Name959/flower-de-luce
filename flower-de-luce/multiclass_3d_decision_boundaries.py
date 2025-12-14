# multiclass_3d_decision_surfaces.py
"""
三分类3D决策边界可视化
使用Iris数据集（三个类别），在后三个特征空间中展示决策边界和淡色半透明的分割面
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def create_multiclass_3d_decision_surfaces():
    """
    创建三分类3D决策边界可视化
    三个分类，后三个特征，一个三维图
    """
    # 1. 加载和准备数据
    print("加载Iris数据集...")
    iris = load_iris()
    X = iris.data  # 所有4个特征
    y = iris.target  # 三个类别
    
    # 选择后三个特征（特征索引1, 2, 3）
    feature_indices = [1, 2, 3]
    X_3d = X[:, feature_indices]
    feature_names = ['Sepal Width', 'Petal Length', 'Petal Width']
    class_names = ['Setosa (Class 0)', 'Versicolor (Class 1)', 'Virginica (Class 2)']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)
    
    # 2. 训练SVM分类器（使用线性核，决策边界是平面）
    print("训练SVM分类器...")
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_scaled, y)
    
    # 3. 创建3D网格用于绘制分割面
    print("创建3D网格...")
    # 确定每个特征的范围
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    z_min, z_max = X_scaled[:, 2].min() - 0.5, X_scaled[:, 2].max() + 0.5
    
    # 创建网格
    grid_resolution = 30
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    
    # 4. 创建可视化
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义颜色
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红色，青色，蓝色
    surface_colors = ['#FFAAAA', '#AAFFEE']  # 淡色版本，只需要两个分割面
    
    # 5. 绘制数据点
    print("绘制数据点...")
    for class_idx in range(3):
        mask = (y == class_idx)
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
                  c=class_colors[class_idx],
                  edgecolors='k',
                  s=100,
                  alpha=0.9,
                  depthshade=True,
                  label=class_names[class_idx])
    
    # 6. 绘制两个关键的分割面（不是三个）
    print("绘制分割面...")
    
    # 对于三个类别，我们只需要两个分割面来分隔它们
    # 我们绘制以下两个分割面：
    # 1. Setosa (0) 和 Versicolor (1) 之间的分割面
    # 2. Versicolor (1) 和 Virginica (2) 之间的分割面
    
    # 第一个分割面：Setosa (0) vs Versicolor (1)
    print("  绘制Setosa和Versicolor之间的分割面...")
    mask_01 = (y == 0) | (y == 1)
    X_01 = X_scaled[mask_01]
    y_01 = y[mask_01]
    y_01_binary = (y_01 == 1).astype(int)  # Setosa=0, Versicolor=1
    
    clf_01 = SVC(kernel='linear', C=1.0, random_state=42)
    clf_01.fit(X_01, y_01_binary)
    
    w_01 = clf_01.coef_[0]
    b_01 = clf_01.intercept_[0]
    
    if abs(w_01[2]) > 1e-10:
        zz_01 = -(w_01[0] * xx + w_01[1] * yy + b_01) / w_01[2]
        
        # 只绘制在数据范围内的部分
        zz_01_masked = np.where((zz_01 >= z_min) & (zz_01 <= z_max), zz_01, np.nan)
        
        # 绘制淡色半透明的分割面
        ax.plot_surface(xx, yy, zz_01_masked,
                      color=surface_colors[0],  # 淡红色
                      alpha=0.3,  # 30%透明度
                      rstride=1, 
                      cstride=1,
                      linewidth=0.5,
                      edgecolor=surface_colors[0],
                      antialiased=True)
    
    # 第二个分割面：Versicolor (1) vs Virginica (2)
    print("  绘制Versicolor和Virginica之间的分割面...")
    mask_12 = (y == 1) | (y == 2)
    X_12 = X_scaled[mask_12]
    y_12 = y[mask_12]
    y_12_binary = (y_12 == 2).astype(int)  # Versicolor=0, Virginica=1
    
    clf_12 = SVC(kernel='linear', C=1.0, random_state=42)
    clf_12.fit(X_12, y_12_binary)
    
    w_12 = clf_12.coef_[0]
    b_12 = clf_12.intercept_[0]
    
    if abs(w_12[2]) > 1e-10:
        zz_12 = -(w_12[0] * xx + w_12[1] * yy + b_12) / w_12[2]
        
        # 只绘制在数据范围内的部分
        zz_12_masked = np.where((zz_12 >= z_min) & (zz_12 <= z_max), zz_12, np.nan)
        
        # 绘制淡色半透明的分割面
        ax.plot_surface(xx, yy, zz_12_masked,
                      color=surface_colors[1],  # 淡青色
                      alpha=0.3,  # 30%透明度
                      rstride=1, 
                      cstride=1,
                      linewidth=0.5,
                      edgecolor=surface_colors[1],
                      antialiased=True)
    
    # 7. 为了更清晰地显示决策区域，添加一些辅助线
    # 在每个分割面的边缘添加轮廓线
    if abs(w_01[2]) > 1e-10:
        # 设置轮廓线的Z值范围
        z_contour = zz_01_masked
        # 移除NaN值
        z_valid = z_contour[~np.isnan(z_contour)]
        if len(z_valid) > 0:
            z_level = np.mean(z_valid)
            ax.contour(xx, yy, zz_01_masked, 
                      levels=[z_level], 
                      colors='red', 
                      linestyles='-',
                      linewidths=1,
                      alpha=0.5,
                      offset=z_level)
    
    if abs(w_12[2]) > 1e-10:
        # 设置轮廓线的Z值范围
        z_contour = zz_12_masked
        # 移除NaN值
        z_valid = z_contour[~np.isnan(z_contour)]
        if len(z_valid) > 0:
            z_level = np.mean(z_valid)
            ax.contour(xx, yy, zz_12_masked, 
                      levels=[z_level], 
                      colors='cyan', 
                      linestyles='-',
                      linewidths=1,
                      alpha=0.5,
                      offset=z_level)
    
    # 8. 设置图形属性
    ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
    ax.set_zlabel(feature_names[2], fontsize=12, fontweight='bold')
    
    ax.set_title('Multiclass 3D Decision Boundaries\n' +
                 'Three Classes with Two Linear Decision Planes', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 设置视角
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
    
    # 调整布局
    plt.tight_layout()
    
    # 9. 显示模型信息
    print("\n" + "="*60)
    print("模型信息:")
    print("="*60)
    print(f"类别: {class_names[0]}, {class_names[1]}, {class_names[2]}")
    print(f"样本数: {len(y)} (Class 0: {sum(y==0)}, Class 1: {sum(y==1)}, Class 2: {sum(y==2)})")
    print(f"特征: {feature_names}")
    print(f"分类器: Linear SVM")
    print(f"分割面数量: 2 (不是3个)")
    print(f"  1. Setosa vs Versicolor (红色)")
    print(f"  2. Versicolor vs Virginica (青色)")
    
    # 计算准确率
    train_accuracy = clf.score(X_scaled, y)
    print(f"训练准确率: {train_accuracy:.2%}")
    print("="*60)
    
    plt.show()
    
    return {
        'X_scaled': X_scaled,
        'y': y,
        'feature_names': feature_names,
        'class_names': class_names,
        'classifier': clf,
        'figure': fig,
        'axes': ax
    }

def create_alternative_visualization():
    """
    替代可视化：使用单个多类SVM绘制决策区域
    """
    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 选择后三个特征
    X_3d = X[:, [1, 2, 3]]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)
    
    # 训练多类SVM
    clf = SVC(kernel='linear', C=1.0, decision_function_shape='ovr', random_state=42)
    clf.fit(X_scaled, y)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    colors = ['red', 'green', 'blue']
    labels = ['Setosa', 'Versicolor', 'Virginica']
    
    for class_idx in range(3):
        mask = (y == class_idx)
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], X_scaled[mask, 2],
                  c=colors[class_idx],
                  edgecolors='k',
                  s=80,
                  alpha=0.9,
                  label=labels[class_idx])
    
    # 绘制两个关键的分割面
    # 使用决策函数的权重
    # 对于一对多(ovr)策略，每个类别有一个决策函数
    # 类别i vs 其他
    
    # 创建网格
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 25),
                        np.linspace(y_min, y_max, 25))
    
    # 绘制第一个分割面（类别0 vs 其他）
    if len(clf.coef_) > 0:
        w0 = clf.coef_[0]  # 类别0 vs 其他
        b0 = clf.intercept_[0]
        
        if abs(w0[2]) > 1e-10:
            zz0 = -(w0[0] * xx + w0[1] * yy + b0) / w0[2]
            ax.plot_surface(xx, yy, zz0,
                           color='lightgray',
                           alpha=0.3,
                           rstride=1, cstride=1,
                           linewidth=0.3)
    
    # 绘制第二个分割面（类别1 vs 其他）
    if len(clf.coef_) > 1:
        w1 = clf.coef_[1]  # 类别1 vs 其他
        b1 = clf.intercept_[1]
        
        if abs(w1[2]) > 1e-10:
            zz1 = -(w1[0] * xx + w1[1] * yy + b1) / w1[2]
            ax.plot_surface(xx, yy, zz1,
                           color='lightblue',
                           alpha=0.3,
                           rstride=1, cstride=1,
                           linewidth=0.3)
    
    # 设置图形属性
    ax.set_xlabel('Sepal Width')
    ax.set_ylabel('Petal Length')
    ax.set_zlabel('Petal Width')
    ax.set_title('Three-Class Decision Boundaries with Two Surfaces')
    ax.legend()
    ax.view_init(elev=25, azim=135)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("三分类3D决策边界可视化")
    print("="*60)
    print("三个分类，后三个特征")
    print("一个三维图包含：")
    print("  - 三个分类的数据点（不同颜色）")
    print("  - 两个淡色半透明的分割面（不是三个）")
    print("="*60)
    
    try:
        results = create_multiclass_3d_decision_surfaces()
        print("\n可视化完成!")
    except Exception as e:
        print(f"主方法出错: {e}")
        print("尝试使用替代可视化...")
        create_alternative_visualization()