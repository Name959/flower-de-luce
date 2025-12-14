import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity
from scipy.special import expit

def create_3d_density_probability_surface():
    iris = load_iris()
    X = iris.data[:, :3]
    y = iris.target
    
    binary_mask = (y == 0) | (y == 1)
    X_binary = X[binary_mask]
    y_binary = y[binary_mask]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_binary)
    
    X_class0 = X_scaled[y_binary == 0]
    X_class1 = X_scaled[y_binary == 1]
    
    bandwidth = 0.5
    kde0 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde0.fit(X_class0[:, :2])
    kde1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde1.fit(X_class1[:, :2])
    
    # 坐标轴范围
    x_min, x_max = -40, 40
    y_min, y_max = -40, 40
    z_min, z_max = -100, 100
    
    grid_resolution = 60
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    
    def smooth_map_to_data_range(grid_values, scale=20.0):
        return np.tanh(grid_values / scale) * 2.0
    
    X_mesh_scaled = smooth_map_to_data_range(X_mesh.ravel())
    Y_mesh_scaled = smooth_map_to_data_range(Y_mesh.ravel())
    grid_points_scaled = np.column_stack([X_mesh_scaled, Y_mesh_scaled])
    
    log_density0 = kde0.score_samples(grid_points_scaled)
    log_density1 = kde1.score_samples(grid_points_scaled)
    
    density0 = np.exp(log_density0)
    density1 = np.exp(log_density1)
    
    eps = 1e-10
    density0 = np.maximum(density0, eps)
    density1 = np.maximum(density1, eps)
    
    # 调整密度值 - 改回2.5
    density0_scaled = density0 * 2.5
    density1_scaled = density1 * 2.5
    density_total = density0_scaled + density1_scaled
    
    # 调整密度权重参数
    min_density_for_certainty = 0.2
    max_density_for_certainty = 1.0
    
    density_weight = np.clip((density_total - min_density_for_certainty) / 
                             (max_density_for_certainty - min_density_for_certainty), 0, 1)
    
    density_ratio = density1_scaled / (density0_scaled + density1_scaled + eps)
    
    # 使用更陡峭的概率转换
    Z_proba = expit((density_ratio - 0.5) * 6) * density_weight + 0.5 * (1 - density_weight)
    Z_proba = np.clip(Z_proba, 0, 1)
    
    Z_mapped = 200 * Z_proba.reshape(X_mesh.shape) - 100
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    Z_proba_reshaped = Z_proba.reshape(X_mesh.shape)
    
    # 绘制主概率曲面 - 缩小范围避免与侧面相接
    shrink_factor = 0.8
    
    # 创建缩小的X和Y网格
    X_mesh_shrunk = X_mesh * shrink_factor
    Y_mesh_shrunk = Y_mesh * shrink_factor
    
    # XOY平面：使用离散颜色圈层 - 完全重新设计
    xy_colors = np.zeros((grid_resolution, grid_resolution, 4))  # RGBA
    
    # 创建清晰的二值分界
    mask_class1 = Z_proba_reshaped > 0.5
    
    # ========== 参考第二个文件的方法：使用缩减颜色范围 ==========
    # 将概率分为5个离散等级
    num_levels = 5
    prob_levels = np.linspace(0, 1, num_levels + 1)
    
    # 定义颜色范围（非常浅的颜色）
    # 绿色（类别1）的颜色范围：从非常浅的绿到中等绿
    green_color_range = {
        'min': [0.92, 1.0, 0.92, 0.8],  # 最浅的绿色
        'max': [0.6, 0.9, 0.6, 0.8]     # 中等绿色
    }
    
    # 红色（类别0）的颜色范围：从非常浅的红到中等红
    red_color_range = {
        'min': [1.0, 0.92, 0.92, 0.8],  # 最浅的红色
        'max': [0.9, 0.6, 0.6, 0.8]     # 中等红色
    }
    
    # 分配颜色 - 参考第二个文件的逻辑
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            prob = Z_proba_reshaped[i, j]
            
            if mask_class1[i, j]:  # 类别1（绿色）
                # 对于绿色区域，概率越高，颜色越深
                # 确定离散等级
                level = 0
                for k in range(num_levels - 1, -1, -1):
                    if prob >= prob_levels[k]:
                        level = k
                        break
                
                # 根据等级插值颜色
                t = level / (num_levels - 1) if num_levels > 1 else 0.5
                min_color = green_color_range['min']
                max_color = green_color_range['max']
                xy_colors[i, j] = [
                    min_color[0] + (max_color[0] - min_color[0]) * t,
                    min_color[1] + (max_color[1] - min_color[1]) * t,
                    min_color[2] + (max_color[2] - min_color[2]) * t,
                    min_color[3]  # alpha保持不变
                ]
            else:  # 类别0（红色）
                # 对于红色区域，概率越低（即1-prob越高），颜色越深
                red_prob = 1 - prob
                # 确定离散等级
                level = 0
                for k in range(num_levels - 1, -1, -1):
                    if red_prob >= prob_levels[k]:
                        level = k
                        break
                
                # 根据等级插值颜色
                t = level / (num_levels - 1) if num_levels > 1 else 0.5
                min_color = red_color_range['min']
                max_color = red_color_range['max']
                xy_colors[i, j] = [
                    min_color[0] + (max_color[0] - min_color[0]) * t,
                    min_color[1] + (max_color[1] - min_color[1]) * t,
                    min_color[2] + (max_color[2] - min_color[2]) * t,
                    min_color[3]  # alpha保持不变
                ]
    # ============================================
    
    # 绘制XOY平面 - 底部平面（离散颜色圈层）使用缩小后的网格
    # 注意：这里使用与概率曲面相同的缩小网格
    ax.plot_surface(
        X_mesh_shrunk,  # 使用缩小后的X网格
        Y_mesh_shrunk,  # 使用缩小后的Y网格
        np.full_like(X_mesh_shrunk, z_min),  # 底部平面
        facecolors=xy_colors,
        shade=False,
        zorder=1,
        rstride=1,
        cstride=1,
        linewidth=0.3,  # 细线
        edgecolor='lightgray',  # 浅灰色边界
        alpha=1.0  # 完全不透明，确保颜色显示
    )
    
    # 绘制主概率曲面（使用缩小后的网格）
    surf = ax.plot_surface(
        X_mesh_shrunk, 
        Y_mesh_shrunk, 
        Z_mapped,
        cmap='viridis',
        alpha=0.85,
        linewidth=0.3,
        edgecolor='black',
        antialiased=True,
        rstride=1,
        cstride=1,
        zorder=10
    )
    
    # ========== 创建真正的投影面 ==========
    # 1. 创建XOZ平面（右侧面）投影
    xoz_x = X_mesh_shrunk  # X坐标保持不变
    xoz_z = Z_mapped  # Z坐标保持不变
    xoz_y = np.full_like(xoz_x, y_max)  # 投影到y=y_max平面
    
    # 为XOZ平面投影着色：使用与概率曲面相同的颜色映射
    xoz_colors = plt.cm.viridis((Z_mapped - z_min) / (z_max - z_min))
    
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
    
    # 2. 创建YOZ平面（左侧面）投影
    yoz_y = Y_mesh_shrunk  # Y坐标保持不变
    yoz_z = Z_mapped  # Z坐标保持不变
    yoz_x = np.full_like(yoz_y, x_min)  # 投影到x=x_min平面
    
    # 为YOZ平面投影着色：使用与概率曲面相同的颜色映射
    yoz_colors = plt.cm.viridis((Z_mapped - z_min) / (z_max - z_min))
    
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
    
    # 添加连接线：从概率曲面到投影面的垂直线
    for i in range(0, grid_resolution, 10):
        for j in range(0, grid_resolution, 10):
            # 从概率曲面上的点到XOZ投影面的线
            ax.plot([X_mesh_shrunk[i, j], X_mesh_shrunk[i, j]], 
                   [Y_mesh_shrunk[i, j], y_max], 
                   [Z_mapped[i, j], Z_mapped[i, j]], 
                   color='gray', alpha=0.2, linewidth=0.3, zorder=4)
            
            # 从概率曲面上的点到YOZ投影面的线
            ax.plot([X_mesh_shrunk[i, j], x_min], 
                   [Y_mesh_shrunk[i, j], Y_mesh_shrunk[i, j]], 
                   [Z_mapped[i, j], Z_mapped[i, j]], 
                   color='gray', alpha=0.2, linewidth=0.3, zorder=4)
    
    # 添加底部投影线（主曲面在Z=z_min处的轮廓线）
    from matplotlib import cm
    contour_levels = np.linspace(z_min, z_max, 10)
    contour = ax.contour(X_mesh_shrunk, Y_mesh_shrunk, Z_mapped, 
                         levels=contour_levels, 
                         zdir='z', 
                         offset=z_min,
                         cmap='viridis',
                         alpha=0.5,
                         linewidths=1.0,
                         zorder=5)
    
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 设置网格样式
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
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
    
    # 设置视角
    ax.view_init(elev=25, azim=225)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Probability Height')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'kde0': kde0,
        'kde1': kde1,
        'X_scaled': X_scaled,
        'y_binary': y_binary,
        'grid': (X_mesh, Y_mesh, Z_mapped),
        'density0': density0,
        'density1': density1,
        'figure': fig,
        'axes': ax
    }

if __name__ == "__main__":
    results = create_3d_density_probability_surface()