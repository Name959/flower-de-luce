# multiclass_3d_probability_map.py
"""
三分类3D概率图可视化
基于二分类概率图(classifier3d.py)进行修改，为三个分类创建3D概率图
其中一个类别下凹，两个类别上凸
使用直接投影法创建两侧投影面
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

def create_multiclass_3d_probability_map():
    """
    创建三分类3D概率图可视化
    三个分类，后三个特征，其中一个类别下凹
    使用直接投影法创建两侧投影面
    地面投影使用缩减颜色范围的方法
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
    class_names = ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_3d)
    
    # 2. 为每个类别创建KDE模型
    print("创建KDE模型...")
    bandwidth = 0.5
    
    # 为每个类别创建KDE（使用前两个特征）
    kdes = []
    for class_idx in range(3):
        X_class = X_scaled[y == class_idx]
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(X_class[:, :2])  # 只使用前两个特征
        kdes.append(kde)
        print(f"  类别 {class_idx}: {len(X_class)} 个样本")
    
    # 3. 创建3D网格
    print("创建3D网格...")
    # 坐标轴范围
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    z_min, z_max = -100, 100
    
    grid_resolution = 60
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    
    # 4. 计算概率并创建映射
    print("计算概率分布...")
    
    def smooth_map_to_data_range(grid_values, scale=20.0):
        """将网格值映射到数据范围"""
        return np.tanh(grid_values / scale) * 2.0
    
    X_mesh_scaled = smooth_map_to_data_range(X_mesh.ravel())
    Y_mesh_scaled = smooth_map_to_data_range(Y_mesh.ravel())
    grid_points_scaled = np.column_stack([X_mesh_scaled, Y_mesh_scaled])
    
    # 计算每个类别的对数密度
    log_densities = []
    for kde in kdes:
        log_density = kde.score_samples(grid_points_scaled)
        log_densities.append(log_density)
    
    # 将对数密度转换为密度
    densities = [np.exp(ld) for ld in log_densities]
    
    # 处理小值
    eps = 1e-10
    densities = [np.maximum(d, eps) for d in densities]
    
    # 调整密度值
    densities_scaled = [d * 2.5 for d in densities]
    density_total = np.sum(densities_scaled, axis=0)
    
    # 调整密度权重参数
    min_density_for_certainty = 0.2
    max_density_for_certainty = 1.0
    
    density_weight = np.clip((density_total - min_density_for_certainty) / 
                             (max_density_for_certainty - min_density_for_certainty), 0, 1)
    
    # 计算每个类别的概率
    probabilities = []
    for class_idx in range(3):
        density_ratio = densities_scaled[class_idx] / (density_total + eps)
        # 使用sigmoid函数转换
        proba = expit((density_ratio - 0.5) * 6) * density_weight + 0.5 * (1 - density_weight)
        proba = np.clip(proba, 0, 1)
        probabilities.append(proba.reshape(X_mesh.shape))
    
    # 5. 将概率映射到Z轴
    print("设置概率映射：一个类别下凹，两个类别上凸...")
    
    Z_mapped_list = []
    
    # 类别0：下凹映射
    Z_class0 = -200 * probabilities[0] + 100  # p=0->100, p=1->-100
    
    # 类别1：上凸映射
    Z_class1 = 200 * probabilities[1] - 100  # p=0->-100, p=1->100
    
    # 类别2：上凸映射
    Z_class2 = 200 * probabilities[2] - 100  # p=0->-100, p=1->100
    
    Z_mapped_list = [Z_class0, Z_class1, Z_class2]
    
    # 找到每个点的最大概率类别和对应的Z值
    prob_array = np.array(probabilities)  # 形状: (3, grid_res, grid_res)
    max_prob = np.max(prob_array, axis=0)
    max_class = np.argmax(prob_array, axis=0)
    
    # 创建组合的Z值：使用最大概率类别对应的Z值
    Z_combined = np.zeros_like(max_prob)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            class_idx = max_class[i, j]
            Z_combined[i, j] = Z_mapped_list[class_idx][i, j]
    
    # 6. 创建可视化图形
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 7. 绘制概率曲面
    print("绘制概率曲面...")
    # 创建缩小的网格，避免与侧面相接
    shrink_factor = 0.8
    
    # 创建缩小的X和Y网格
    X_mesh_shrunk = X_mesh * shrink_factor
    Y_mesh_shrunk = Y_mesh * shrink_factor
    
    # 为曲面创建颜色 - 使用更浅的颜色
    facecolors = np.zeros((grid_resolution, grid_resolution, 4))
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            class_idx = max_class[i, j]
            prob_value = max_prob[i, j]
            
            # 设置颜色和透明度 - 使用更浅的颜色
            if class_idx == 0:  # 红色（下凹）
                # 对于下凹类别，概率越高，颜色越深
                alpha = 0.3 + prob_value * 0.4  # 降低透明度变化范围
                # 使用更浅的红色
                facecolors[i, j] = [1.0, 0.92, 0.92, alpha]
            elif class_idx == 1:  # 绿色（上凸）
                alpha = 0.3 + prob_value * 0.4
                # 使用更浅的绿色
                facecolors[i, j] = [0.92, 1.0, 0.92, alpha]
            else:  # 蓝色（上凸）
                alpha = 0.3 + prob_value * 0.4
                # 使用更浅的蓝色
                facecolors[i, j] = [0.92, 0.92, 1.0, alpha]
    
    # 绘制概率曲面
    surf = ax.plot_surface(
        X_mesh_shrunk, 
        Y_mesh_shrunk, 
        Z_combined,
        facecolors=facecolors,
        alpha=0.7,
        linewidth=0.3,
        edgecolor='black',
        antialiased=True,
        rstride=1,
        cstride=1,
        zorder=10
    )
    
    # 8. 创建投影面
    print("创建投影面...")
    
    # 8.1 XOY平面（底部）投影 - 使用缩减颜色范围的方法
    xy_colors = np.zeros((grid_resolution, grid_resolution, 4))
    
    # ========== 修改：使用缩减颜色范围的方法 ==========
    # 定义每个类别的颜色范围
    # 红色（类别0）：从浅红到深红
    red_color_range = {
        'min': [1.0, 0.92, 0.92, 0.7],  # 最浅的红色
        'max': [1.0, 0.7, 0.7, 0.7]     # 最深的红色
    }
    
    # 绿色（类别1）：从浅绿到深绿
    green_color_range = {
        'min': [0.92, 1.0, 0.92, 0.7],  # 最浅的绿色
        'max': [0.7, 1.0, 0.7, 0.7]     # 最深的绿色
    }
    
    # 蓝色（类别2）：从浅蓝到深蓝
    blue_color_range = {
        'min': [0.92, 0.92, 1.0, 0.7],  # 最浅的蓝色
        'max': [0.7, 0.7, 1.0, 0.7]     # 最深的蓝色
    }
    
    # 将概率分为离散等级
    num_levels = 5
    prob_levels = np.linspace(0, 1, num_levels + 1)
    
    # 分配颜色
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            class_idx = max_class[i, j]
            prob_value = max_prob[i, j]
            
            # 确定离散等级
            level = 0
            for k in range(num_levels - 1, -1, -1):
                if prob_value >= prob_levels[k]:
                    level = k
                    break
            
            # 根据等级插值颜色
            t = level / (num_levels - 1) if num_levels > 1 else 0.5
            
            if class_idx == 0:  # 红色
                min_color = red_color_range['min']
                max_color = red_color_range['max']
                xy_colors[i, j] = [
                    min_color[0] + (max_color[0] - min_color[0]) * t,
                    min_color[1] + (max_color[1] - min_color[1]) * t,
                    min_color[2] + (max_color[2] - min_color[2]) * t,
                    min_color[3]  # alpha保持不变
                ]
            elif class_idx == 1:  # 绿色
                min_color = green_color_range['min']
                max_color = green_color_range['max']
                xy_colors[i, j] = [
                    min_color[0] + (max_color[0] - min_color[0]) * t,
                    min_color[1] + (max_color[1] - min_color[1]) * t,
                    min_color[2] + (max_color[2] - min_color[2]) * t,
                    min_color[3]  # alpha保持不变
                ]
            else:  # 蓝色
                min_color = blue_color_range['min']
                max_color = blue_color_range['max']
                xy_colors[i, j] = [
                    min_color[0] + (max_color[0] - min_color[0]) * t,
                    min_color[1] + (max_color[1] - min_color[1]) * t,
                    min_color[2] + (max_color[2] - min_color[2]) * t,
                    min_color[3]  # alpha保持不变
                ]
    # ============================================
    
    # 绘制XOY平面
    ax.plot_surface(
        X_mesh_shrunk, Y_mesh_shrunk, np.full_like(X_mesh_shrunk, z_min),
        facecolors=xy_colors,
        shade=False,
        zorder=1,
        rstride=1,
        cstride=1,
        linewidth=0.3,  # 减少线宽
        edgecolor='lightgray',  # 使用浅灰色边界
        alpha=0.9  # 增加透明度
    )
    
    # 8.2 XOZ平面（右侧面）投影 - 使用直接投影法
    # 投影方法：将概率曲面上的点沿着Y轴方向投影到XOZ平面
    xoz_x = X_mesh_shrunk  # X坐标保持不变
    xoz_z = Z_combined  # Z坐标保持不变
    xoz_y = np.full_like(xoz_x, y_max * shrink_factor)  # 投影到y=y_max*shrink_factor平面
    
    # 为XOZ平面投影着色：使用与概率曲面相同的颜色
    xoz_colors = facecolors
    
    # 绘制XOZ平面投影
    ax.plot_surface(
        xoz_x, xoz_y, xoz_z,
        facecolors=xoz_colors,
        shade=False,
        zorder=5,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        edgecolor='none',
        alpha=0.6
    )
    
    # 8.3 YOZ平面（左侧面）投影 - 使用直接投影法
    # 投影方法：将概率曲面上的点沿着X轴方向投影到YOZ平面
    yoz_y = Y_mesh_shrunk  # Y坐标保持不变
    yoz_z = Z_combined  # Z坐标保持不变
    yoz_x = np.full_like(yoz_y, x_min * shrink_factor)  # 投影到x=x_min*shrink_factor平面
    
    # 为YOZ平面投影着色：使用与概率曲面相同的颜色
    yoz_colors = facecolors
    
    # 绘制YOZ平面投影
    ax.plot_surface(
        yoz_x, yoz_y, yoz_z,
        facecolors=yoz_colors,
        shade=False,
        zorder=5,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        edgecolor='none',
        alpha=0.6
    )
    
    # 9. 添加连接线：从概率曲面到投影面的垂直线
    print("添加连接线...")
    for i in range(0, grid_resolution, 10):
        for j in range(0, grid_resolution, 10):
            # 从概率曲面上的点到XOZ投影面的线
            ax.plot([X_mesh_shrunk[i, j], X_mesh_shrunk[i, j]], 
                   [Y_mesh_shrunk[i, j], y_max * shrink_factor], 
                   [Z_combined[i, j], Z_combined[i, j]], 
                   color='gray', alpha=0.2, linewidth=0.3, zorder=4)
            
            # 从概率曲面上的点到YOZ投影面的线
            ax.plot([X_mesh_shrunk[i, j], x_min * shrink_factor], 
                   [Y_mesh_shrunk[i, j], Y_mesh_shrunk[i, j]], 
                   [Z_combined[i, j], Z_combined[i, j]], 
                   color='gray', alpha=0.2, linewidth=0.3, zorder=4)
    
    # 10. 添加底部轮廓线
    from matplotlib import cm
    contour_levels = np.linspace(z_min, z_max, 10)
    contour = ax.contour(X_mesh_shrunk, Y_mesh_shrunk, Z_combined, 
                         levels=contour_levels, 
                         zdir='z', 
                         offset=z_min,
                         cmap='viridis',
                         alpha=0.5,
                         linewidths=1.0,
                         zorder=5)
    
    # 11. 设置图形属性
    ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
    ax.set_zlabel('Probability Mapped to [-100, 100]', fontsize=12, fontweight='bold')
    
    ax.set_title('Three-Class 3D Probability Map with Side Projections\n' +
                 'Class 0: Concave Down (Red)\n' +
                 'Classes 1 & 2: Convex Up (Green & Blue)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 设置视角
    ax.view_init(elev=25, azim=225)
    
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
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    
    # 12. 添加颜色条
    # 创建颜色映射
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(Z_combined)
    
    # 添加颜色条
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Probability Mapped to [-100, 100]', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 13. 显示模型信息
    print("\n" + "="*60)
    print("模型信息:")
    print("="*60)
    print(f"类别: {class_names[0]}, {class_names[1]}, {class_names[2]}")
    print(f"样本数: {len(y)} (Class 0: {sum(y==0)}, Class 1: {sum(y==1)}, Class 2: {sum(y==2)})")
    print(f"特征: {feature_names}")
    print(f"方法: Kernel Density Estimation (KDE)")
    print(f"带宽: {bandwidth}")
    print(f"密度缩放: 2.5")
    print(f"Z轴范围: {z_min} 到 {z_max}")
    print(f"概率映射:")
    print(f"  - Class 0 (Red): 下凹，概率1映射到-100，概率0映射到100")
    print(f"  - Class 1 (Green): 上凸，概率1映射到100，概率0映射到-100")
    print(f"  - Class 2 (Blue): 上凸，概率1映射到100，概率0映射到-100")
    print(f"地面投影: 缩减颜色范围法（每个类别有5个颜色等级）")
    print(f"颜色范围:")
    print(f"  - 红色: 从 [1.0, 0.92, 0.92] 到 [1.0, 0.7, 0.7]")
    print(f"  - 绿色: 从 [0.92, 1.0, 0.92] 到 [0.7, 1.0, 0.7]")
    print(f"  - 蓝色: 从 [0.92, 0.92, 1.0] 到 [0.7, 0.7, 1.0]")
    print(f"投影方法: 直接投影法（每个点投影到侧面）")
    print(f"网格分辨率: {grid_resolution}×{grid_resolution}")
    print(f"缩小因子: {shrink_factor}")
    print("="*60)
    
    plt.show()
    
    return {
        'X_scaled': X_scaled,
        'y': y,
        'kdes': kdes,
        'probabilities': probabilities,
        'Z_mapped_list': Z_mapped_list,
        'Z_combined': Z_combined,
        'figure': fig,
        'axes': ax
    }

if __name__ == "__main__":
    print("="*60)
    print("三分类3D概率图可视化（带侧面投影）")
    print("="*60)
    print("三个分类，后三个特征")
    print("一个类别下凹，两个类别上凸")
    print("一张图包含：")
    print("  - 立体概率图（三个类别）")
    print("  - 三个投影面（xoy, xoz, yoz）使用直接投影法")
    print("  - 概率映射到-100到100范围")
    print("  - 地面投影使用缩减颜色范围法")
    print("  - 连接线显示投影关系")
    print("="*60)
    
    try:
        results = create_multiclass_3d_probability_map()
        print("\n可视化完成!")
    except Exception as e:
        print(f"主方法出错: {e}")
        import traceback
        traceback.print_exc()