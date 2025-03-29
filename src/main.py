import time
from pathlib import Path

import pandas as pd
from dstz.math.stat.moment import information_content, high_order_moment

from cf.generation import sin_p1, inf_content
from cf.group import FMassDistribution

# 定义采样基数范围，从 2 到 14
cards = [i for i in range(2, 15)]
# 定义高阶矩的阶数范围，从 1 到 9
orders = [i for i in range(1, 10)]
# 定义生成函数列表，这里仅使用 sin_p1 函数
gs = [sin_p1]

# 用于存储每次计算结果的数据列表
data = []

# 遍历生成函数列表
for g in gs:
    # 遍历采样基数范围
    for card in cards:
        # 遍历高阶矩的阶数范围
        for order in orders:
            print("_" * 20)
            print(f"g: {g.__name__}, card: {card}, order: {order}")
            # 创建 FMassDistribution 实例
            fm = FMassDistribution(g)
            # 进行采样，获取向量和质量分布
            vector, mass = fm.sampling(card)
            # 记录开始时间
            start_h = time.time()

            # 再次创建 FMassDistribution 实例，这里可考虑优化避免重复创建
            fm = FMassDistribution(sin_p1)
            # 再次进行采样，获取向量和质量分布
            vector, mass = fm.sampling(card)

            # 记录特征函数计算开始时间
            start_cf = time.time()
            # 使用特征函数法计算高阶矩
            cf = fm.high_order_moments_from_cf(order, vector, inf_content)
            # 记录特征函数计算结束时间
            end_cf = time.time()
            # 计算特征函数计算耗时
            cf_time = end_cf - start_cf

            # 记录直接计算高阶矩的开始时间
            start_h = time.time()
            # 直接计算高阶矩
            h = high_order_moment(mass, information_content, order)
            # 记录直接计算高阶矩的结束时间
            end_h = time.time()
            # 计算直接计算高阶矩的耗时
            h_time = end_h - start_h

            print(f"特征函数结果: {cf:.4f}, 直接求结果: {h:.4f}")
            print(f"特征函数耗时: {cf_time:.4f}秒, 直接求耗时: {h_time:.4f}秒")

            # 将每次计算结果添加到数据列表中
            data.append([g.__name__, card, order, f"{cf:.4f}", f"{h:.4f}", f"{cf_time:.4f}", f"{h_time:.4f}"])

# 将数据列表转换为 Pandas 的 DataFrame
df = pd.DataFrame(data, columns=["生成函数", "FOD", "阶数", "特征函数法", "直接法", "特征函数法时间", "直接法时间"])
# 定义输出路径
path = Path("./out")
# 创建输出目录，如果目录已存在则不报错
path.mkdir(parents=True, exist_ok=True)
# 将 DataFrame 保存为 CSV 文件
df.to_csv("./out/data.csv", index=False)
