# This code is part of LINKEQ.
#
# (C) Copyright LINKE 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# -*- coding: utf-8 -*-
# @Time     : 2024/2/29 12:31
# @Author   : HFLASH @ LINKE
# @File     : project_structure_generator.py
# @Software : PyCharm


# -*- coding: utf-8 -*-
# @Author : Frica01
# @Time   : 2023-06-22 0:00
# @Name   : project_structure_generator.py

import os

# 需要排除的文件夹列表
exclude_folders = ['Scratches and Consoles','External Libraries','optimizing-cd-protocols','.git', '.idea', '.vscode', '.venv', '__pycache__', 'project_structure_generator.py']
def generate_project_structure(directory, indent='', is_last=False, is_root=False, output_file=None):
    """
    生成项目结构的文字样式

    Args:
        directory (str): 目录路径
        indent (str): 缩进字符串
        is_last (bool): 是否是最后一个元素
        is_root (bool): 是否是根目录
        output_file (file): 输出文件对象

    Returns:
        None
    """
    # 获取目录中的文件和文件夹列表，并按照一定的规则排序

    items = sorted(os.listdir(directory),
                   key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x != '__init__.py', x))
    num_items = len(items)

    if is_root:
        # 根目录名称
        output_file.write(f"{os.path.basename(os.getcwd())}/\n")

    for i, item in enumerate(items):
        if item in exclude_folders:
            continue

        item_path = os.path.join(directory, item)
        is_item_last = i == num_items - 1

        if os.path.isdir(item_path):
            # 如果是目录，则添加目录标记并递归生成目录结构
            marker = '└── ' if is_item_last else '├── '
            output_file.write(f"{indent}{marker}{item}/\n")
            new_indent = indent + ('    ' if is_last else '│   ')
            generate_project_structure(item_path, new_indent, is_item_last, output_file=output_file)
        else:
            # 如果是文件，则添加文件标记
            marker = '└── ' if is_item_last else '├── '
            output_file.write(f"{indent}{marker}{item}\n")
if __name__ == '__main__':
    # 打开要写入的文件
    with open('project_structure.txt', 'w') as file:
        # 生成项目结构并写入文件
        generate_project_structure('.', is_root=True, output_file=file)

    print("目录结构已写入文件 project_structure.txt")
