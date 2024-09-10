import os
import random
import shutil

# 假设这四个目录路径已知
source_dir1 = '/home/libo/Dual-ArbNet/demodata/val_mattest'
# source_dir2 = '/home/libo/Dual-ArbNet/demodata/ref_mat'
dest_dir1 = '/home/libo/Dual-ArbNet/demodata/mattest'
# dest_dir2 = '/home/libo/Dual-ArbNet/val/ref_mat'

# 获取源目录下的所有文件名
files1 = sorted(os.listdir(source_dir1))
# files2 = sorted(os.listdir(source_dir2))

# 计算每个目录应选择的文件数
num_files1 = len(files1)
# num_files2 = len(files2)
num_files_to_select = num_files1 // 2  # 选取五分之一的文件对

# 随机选择文件索引
selected_indices = random.sample(range(num_files1), num_files_to_select)

# 根据选择的索引，得到文件名列表
selected_files1 = [files1[idx] for idx in selected_indices]
# selected_files2 = [files2[idx] for idx in selected_indices]

# 复制文件到目标目录
for file1 in selected_files1:
    source_file1 = os.path.join(source_dir1, file1)
    # source_file2 = os.path.join(source_dir2, file2)
    
    dest_file1 = os.path.join(dest_dir1, file1)
    # dest_file2 = os.path.join(dest_dir2, file2)
    
    shutil.move(source_file1, dest_file1)
    # shutil.move(source_file2, dest_file2)


