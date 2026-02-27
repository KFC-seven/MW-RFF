import os
import shutil

# 配置路径
source_root = "F:\MW-RFF\IQ_signal_plots"  # 包含所有原始数据的文件夹
target_root = "E:\program\IQ_signal_plots"  # 包含150个子文件夹的路径
copy_folder = "trajectory_pos"  # 要复制的文件夹名（如 "data"）

# 遍历源文件夹下的所有子文件夹
for subdir in os.listdir(source_root):
    source_sub_path = os.path.join(source_root, subdir)
    target_sub_path = os.path.join(target_root, subdir)
    
    # 检查源子文件夹中是否存在要复制的文件夹
    source_folder = os.path.join(source_sub_path, copy_folder)
    if not os.path.isdir(source_folder):
        print(f"跳过: 源文件夹中未找到 {subdir}/{copy_folder}")
        continue
    
    # 创建目标子文件夹（如果不存在）
    os.makedirs(target_sub_path, exist_ok=True)
    
    # 目标文件夹路径（如 target_root/subdir/XXX）
    target_folder = os.path.join(target_sub_path, copy_folder)
    
    # 如果目标已存在同名文件夹，合并内容（覆盖同名文件）
    if os.path.exists(target_folder):
        # 递归复制并覆盖已有文件
        for item in os.listdir(source_folder):
            src = os.path.join(source_folder, item)
            dst = os.path.join(target_folder, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)  # Python 3.8+ 支持合并
            else:
                shutil.copy2(src, dst)  # 保留文件元数据
        print(f"已合并: {subdir}/{copy_folder}")
    else:
        # 直接复制整个文件夹
        shutil.copytree(source_folder, target_folder)
        print(f"已复制: {subdir}/{copy_folder}")