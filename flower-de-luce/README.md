# 3D 决策边界与分类器可视化示例

本仓库包含用于可视化分类器决策边界与概率分布的 Python 脚本（2D/3D）。主要用于教学、实验与比较不同分类器在合成或真实数据上的表现。

**仓库结构**
- `main.py`：演示入口（若存在），可作为运行示例。
- `classifier2d.py`：2D 分类器可视化示例（决策边界、数据点）。
- `compare_classifiers.py`：比较多个分类器性能与边界的脚本。
- `data_preview.py`：用于快速查看/绘制数据集分布的工具脚本。
- `binary_3d_decision_boundary.py`：二分类 3D 决策面可视化脚本。
- `binary_3d_probability_map.py`：二分类 3D 概率图（概率场）可视化脚本。
- `multiclass_3d_decision_boundaries.py`：多分类 3D 决策面可视化。
- `multiclass_3d_probability_map.py`：多分类 3D 概率图可视化。

**运行环境与依赖**
建议使用 Python 3.8+（仓库在 Python 3.10/3.11 下也应可运行）。主要依赖：

- `numpy`
- `scikit-learn`
- `matplotlib`
- `plotly`（可选，若脚本使用交互式 3D 可视化）

快速安装依赖示例：
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install numpy scikit-learn matplotlib plotly
```

如果你偏好将依赖写入 `requirements.txt`，可以运行：
```powershell
pip freeze > requirements.txt
```

**使用示例**
在激活虚拟环境后，可直接运行需要的脚本。例如：
```powershell
python main.py
python classifier2d.py
python binary_3d_decision_boundary.py
```

某些脚本可能会在运行时弹出图形窗口或在终端输出简要信息，请根据需要调整脚本中参数（通常在文件顶部或 `if __name__ == '__main__':` 区域）。

**提示**
- 若要保存可视化输出为图片或交互式 HTML，请查看脚本中 `save`/`show` 类参数的实现并按需修改。
- 若脚本报错找不到模块，请确认虚拟环境已激活且依赖已安装。

---
如需我为 `requirements.txt` 自动生成依赖清单、添加运行示例或完善 `main.py` 启动逻辑，我可以继续帮你处理。是否需要我继续？
