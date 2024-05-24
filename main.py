structure = {
    "1_应用数学与机器学习基础": {
        "2_线性代数": [
            "标量、向量、矩阵和张量",
            "矩阵和向量相乘",
            "单位矩阵和逆矩阵",
            "线性相关和生成子空间",
            "范数",
            "特殊类型的矩阵和向量",
            "特征分解",
            "奇异值分解",
            "Moore-Penrose 伪逆",
            "迹运算",
            "行列式",
            "实例：主成分分析"
        ],
        "3_概率与信息论": [
            "为什么要使用概率",
            "随机变量",
            "概率分布",
            "离散型变量和概率质量函数",
            "连续型变量和概率密度函数",
            "边缘概率",
            "条件概率",
            "条件概率的链式法则",
            "独立性和条件独立性",
            "期望、方差和协方差",
            "常用概率分布",
            "Bernoulli 分布",
            "Multinoulli 分布",
            "高斯分布",
            "指数分布和 Laplace 分布",
            "Dirac 分布和经验分布",
            "分布的混合",
            "常用函数的有用性质",
            "贝叶斯规则",
            "连续型变量的技术细节",
            "信息论",
            "结构化概率模型"
        ],
        "4_数值计算": [
            "上溢和下溢",
            "病态条件",
            "基于梯度的优化方法",
            "梯度之上：Jacobian 和 Hessian 矩阵",
            "约束优化",
            "实例：线性最小二乘"
        ],
        "5_机器学习基础": [
            "学习算法",
            "任务 T",
            "性能度量 P",
            "经验 E",
            "示例：线性回归",
            "容量、过拟合和欠拟合",
            "没有免费午餐定理",
            "正则化",
            "超参数和验证集",
            "交叉验证",
            "估计、偏差和方差",
            "点估计",
            "偏差",
            "方差和标准差",
            "权衡偏差和方差以最小化均方误差",
            "一致性",
            "最大似然估计",
            "条件对数似然和均方误差",
            "最大似然的性质",
            "贝叶斯统计",
            "最大后验 (MAP) 估计",
            "监督学习算法",
            "概率监督学习",
            "支持向量机",
            "其他简单的监督学习算法",
            "无监督学习算法",
            "主成分分析",
            "k-均值聚类",
            "随机梯度下降",
            "构建机器学习算法",
            "促使深度学习发展的挑战",
            "维数灾难",
            "局部不变性和平滑正则化",
            "流形学习"
        ]
    },
    "2_深度网络：现代实践": {
        "6_深度前馈网络": [
            "实例：学习 XOR",
            "基于梯度的学习",
            "代价函数",
            "输出单元",
            "隐藏单元",
            "整流线性单元及其扩展",
            "logistic sigmoid 与双曲正切函数",
            "其他隐藏单元",
            "架构设计",
            "万能近似性质和深度",
            "其他架构上的考虑",
            "反向传播和其他的微分算法",
            "计算图",
            "微积分中的链式法则",
            "递归地使用链式法则来实现反向传播",
            "全连接 MLP 中的反向传播计算",
            "符号到符号的导数",
            "一般化的反向传播",
            "实例：用于 MLP 训练的反向传播",
            "复杂化",
            "深度学习界以外的微分",
            "高阶微分",
            "历史小记"
        ],
        "7_深度学习中的正则化": [
            "参数范数惩罚",
            "L2 参数正则化",
            "L1 正则化",
            "作为约束的范数惩罚",
            "正则化和欠约束问题",
            "数据集增强",
            "噪声鲁棒性",
            "向输出目标注入噪声",
            "半监督学习",
            "多任务学习",
            "提前终止",
            "参数绑定和参数共享",
            "卷积神经网络",
            "稀疏表示",
            "Bagging 和其他集成方法",
            "Dropout",
            "对抗训练",
            "切面距离、正切传播和流形正切分类器"
        ]
    }
}

import os
import json
from typing import Union, Dict, List, Any

def create_directories_and_files(
    base_path: str, 
    structure: Dict[str, Any], 
    readme_file, 
    parent_path: str = "", 
    level: int = 1
):
    """
    根据给定的目录结构创建目录和文件，并生成 README.md 文件。

    Args:
        base_path (str): 根目录路径。
        structure (Dict[str, Any]): 目录结构的嵌套字典。
        readme_file (File): 用于写入README内容的文件对象。
        parent_path (str): 父目录路径。
        level (int): 目录的层级，用于确定 README 标题级别。

    Returns:
        None
    """
    heading = "#" * level

    for key, value in structure.items():
        current_path = os.path.join(base_path, key.replace(" ", "_").replace("-", "_"))

        # 创建目录
        os.makedirs(current_path, exist_ok=True)

        # 在README中添加章节标题
        if parent_path:
            readme_file.write(f"{heading} {parent_path}/{key}\n\n")
        else:
            readme_file.write(f"{heading} {key}\n\n")

        # 递归调用创建子目录和文件
        if isinstance(value, dict) and value:
            create_directories_and_files(
                current_path, 
                value, 
                readme_file, 
                parent_path + "/" + key if parent_path else key, 
                level + 1
            )
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                item = f"{idx:02d}_{item}"
                file_name = item.replace(" ", "_").replace("-", "_") + ".py"
                file_path = os.path.join(current_path, file_name)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f"# {item}\n\n")
                    file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                # 在README中添加文件链接
                item_clean = item.replace(" ", "_").replace("-", "_")
                readme_file.write(f"- [{item}](./{parent_path}/{key}/{item_clean}.py)\n")
        else:
            # 创建文件并写入初始内容
            file_name = key.replace(" ", "_").replace("-", "_") + ".py"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")

        # 添加空行以分隔不同的章节
        readme_file.write("\n")

def main():
    root_dir = './'
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建 README.md 文件
    with open(os.path.join(root_dir, "README.md"), 'w', encoding='utf-8') as readme_file:
        readme_file.write("# 深度学习\n\n")
        readme_file.write("这是一个关于深度学习的目录结构。\n\n")
        create_directories_and_files(root_dir, structure, readme_file)

    print("目录和文件结构已生成，并创建 README.md 文件。")

if __name__ == "__main__":
    main()
