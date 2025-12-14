#!/usr/bin/env python3
"""
3D分类器可视化工具集 - 主程序
包含多种3D可视化工具，用于展示分类器的概率分布和决策边界
"""

import sys
import os

def print_header():
    """打印程序标题"""
    print("=" * 70)
    print("              3D分类器可视化工具集")
    print("=" * 70)
    print()

def print_menu():
    """打印主菜单"""
    print("请选择要运行的可视化工具：")
    print("=" * 40)
    print("1. 二分类3D概率图")
    print("   基于KDE的3D概率曲面，带有投影面和连接线")
    print()
    print("2. 三分类3D概率图")  
    print("   三个类别的3D概率曲面，一个下凹两个上凸，带有投影面")
    print()
    print("3. 二分类3D决策边界")
    print("   线性SVM的3D决策边界，带有半透明分割面")
    print()
    print("4. 分类器比较可视化")
    print("   多个分类器的决策边界和概率对比（2D）")
    print()
    print("5. 数据预览和探索")
    print("   Iris数据集的可视化探索")
    print()
    print("6. 运行所有可视化")
    print("   依次运行所有可视化工具")
    print()
    print("0. 退出程序")
    print("=" * 40)

def run_binary_3d_probability():
    """运行二分类3D概率图"""
    print("\n正在加载二分类3D概率图...")
    try:
        from binary_3d_probability_map import create_3d_density_probability_surface
        results = create_3d_density_probability_surface()
        print("✓ 二分类3D概率图完成！")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_multiclass_3d_probability():
    """运行三分类3D概率图"""
    print("\n正在加载三分类3D概率图...")
    try:
        from multiclass_3d_probability_map import create_multiclass_3d_probability_map
        results = create_multiclass_3d_probability_map()
        print("✓ 三分类3D概率图完成！")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_binary_3d_decision_boundary():
    """运行二分类3D决策边界"""
    print("\n正在加载二分类3D决策边界...")
    try:
        from binary_3d_decision_boundary import create_binary_3d_decision_surface
        results = create_binary_3d_decision_surface()
        print("✓ 二分类3D决策边界完成！")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_compare_classifiers():
    """运行分类器比较可视化"""
    print("\n正在加载分类器比较可视化...")
    try:
        from compare_classifiers import *
        print("✓ 分类器比较可视化完成！")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_data_preview():
    """运行数据预览和探索"""
    print("\n正在加载数据预览和探索...")
    try:
        from data_preview import *
        print("✓ 数据预览和探索完成！")
    except Exception as e:
        print(f"✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_all():
    """运行所有可视化"""
    run_binary_3d_probability()
    run_multiclass_3d_probability()
    run_binary_3d_decision_boundary()
    run_compare_classifiers()
    run_data_preview()

def main():
    """主程序"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("\n请输入选择 (0-6): ").strip()
            
            if choice == "0":
                print("\n感谢使用3D分类器可视化工具集！")
                break
            elif choice == "1":
                run_binary_3d_probability()
            elif choice == "2":
                run_multiclass_3d_probability()
            elif choice == "3":
                run_binary_3d_decision_boundary()
            elif choice == "4":
                run_compare_classifiers()
            elif choice == "5":
                run_data_preview()
            elif choice == "6":
                print("\n开始运行所有可视化...")
                print("=" * 50)
                run_all()
                print("=" * 50)
                print("所有可视化已完成！")
            else:
                print("无效选择，请重新输入！")
                continue
            
            # 询问是否继续
            if choice != "0":
                cont = input("\n是否继续使用其他工具？ (y/n): ").strip().lower()
                if cont not in ['y', 'yes']:
                    print("\n感谢使用3D分类器可视化工具集！")
                    break
        
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            cont = input("\n是否继续？ (y/n): ").strip().lower()
            if cont not in ['y', 'yes']:
                break

if __name__ == "__main__":
    # 检查依赖
    required_packages = ['numpy', 'matplotlib', 'scikit-learn', 'scipy', 'pandas', 'seaborn']
    
    print("正在检查依赖包...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 未安装")
            print(f"  请运行: pip install {package}")
    
    print("\n" + "=" * 50)
    main()