from os.path import splitext
import os

# 两个目录的路径
dir1 = './demodata/mattest'
dir2 = './demodata/ref_mat'

# 获取目录中的文件名列表
files1 = sorted(os.listdir(dir1))  # 假设文件已经按顺序排列
files2 = sorted(os.listdir(dir2))  # 假设文件已经按顺序排列

# 确保两个目录中的文件数量相同
# if len(files1) != len(files2):
#     raise ValueError("Directories must contain the same number of files.")
print(len(files1))
print(len(files2))
# 创建一个txt文件来存储结果
output_file = 'demodata/multiname.txt'

with open(output_file, 'w') as f:
    # 将每对文件名写入txt文件中
    for file1, file2 in zip(files1, files2):
        if file1.replace('T2', 'PD') != file2:
            print("file1 dont match file2: file1:",file1, " file2:", file2)
            continue
        # print("file1 match file2: file1:",file1, " file2:", file2)
        file1_name = splitext(file1)[0]
        file2_name = splitext(file2)[0]
        f.write(f"{file1_name}\t{file2_name}\n")

print(f"文件名对已写入到 {output_file}")
