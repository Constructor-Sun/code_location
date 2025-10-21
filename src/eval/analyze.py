import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib import cm
from typing import List, Optional

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def to_percentages_exact(values):
    percentages = [round(v * 100, 2) for v in values]
    
    total = sum(percentages)
    diff = 100 - total
    
    if abs(diff) > 0.001:
        max_index = percentages.index(max(percentages))
        percentages[max_index] = round(percentages[max_index] + diff, 2)
    
    return percentages

def plot_venn_simple(proportion, 
                     ref_label,
                     candidate_label, 
                     title,
                     savepath="plots/pic.png"):
    plt.figure(figsize=(8, 6))
    
    # 使用Paired配色方案
    paired_colors = cm.get_cmap('Paired', 12)
    
    # 绘制维恩图
    venn = venn2(subsets=to_percentages_exact(proportion), 
                 set_labels=(ref_label, candidate_label),
                 set_colors=(paired_colors(6), paired_colors(12)),
                 alpha=0.9)
    
    # 自定义颜色
    if venn.get_label_by_id('10'):
        venn.get_patch_by_id('10').set_color(paired_colors(1))  # A only
    if venn.get_label_by_id('01'):
        venn.get_patch_by_id('01').set_color(paired_colors(4))  # B only
    if venn.get_label_by_id('11'):
        venn.get_patch_by_id('11').set_color(paired_colors(8))  # Intersection
    
    for label in venn.set_labels:
        if label:
            label.set_fontsize(24)
    
    for label in venn.subset_labels:
        if label:
            label.set_fontsize(24)

    plt.setp(venn.patches, linewidth=1, edgecolor='black', linestyle='-')
    # plt.title(title, fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def create_academic_pie_chart(
    proportions: List[float],
    labels: List[str],
    filename: str,
    title: Optional[str] = 'Proportion Distribution',
    colormap_name: str = "Paired", # "Paired", # "Blues", # 'Pastel1' # can also use customized colors
    dpi: int = 300
):
    cmap = plt.cm.get_cmap(colormap_name)
    colors = cmap(np.linspace(0.1, 0.7, len(proportions))) 

    fig, ax = plt.subplots(figsize=(8, 8)) 
    wedges, texts, autotexts = ax.pie(
        proportions, 
        labels=labels, 
        autopct='%1.1f%%',       # 显示百分比，保留一位小数
        startangle=90,           # 从顶部开始
        colors=colors,           # 专业配色
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5} # 黑色边框，增强区分度
    )

    plt.setp(autotexts, size=24, weight="bold", color="black")
    plt.setp(texts, size=24, weight="bold")
    # ax.set_title(title, fontsize=24, fontweight='bold')
    ax.axis('equal')  

    plt.tight_layout() 
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def diff_and_proportion(candidate, ref):
    ref_unique = ref - candidate
    candidate_unique = candidate - ref
    common = ref & candidate
    ref_proportion = len(ref_unique) / len(candidate | ref)
    common_proportion = len(common) / len(candidate | ref)
    candidate_proportion = len(candidate_unique) / len(candidate | ref)
    return ref_unique, candidate_unique, [ref_proportion, candidate_proportion, common_proportion]

def main():
    json_file_path = "datasets/swe-bench-lite-subset-summary.json"
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    ref = set(data_dict["SweRank-32B"])
    qwen_30B = set(data_dict["Qwen3-Coder-30B"])
    qwen_480B = set(data_dict["Qwen3-Coder-480B"])
    xai = set(data_dict["Grok-Code-Fast-1"])

    intersections = ref & qwen_30B & qwen_480B & xai
    print("intersections length: ", len(intersections))
    print("intersections: ", intersections)
    ref = ref - intersections
    qwen_30B = qwen_30B - intersections
    qwen_480B = qwen_480B - intersections
    xai = xai - intersections

    ref_unique, qwen_30B_unique, qwen_30B_proportion = diff_and_proportion(qwen_30B, ref)
    print("ref_unique: ", ref_unique)
    print("qwen_30B_unique: ", qwen_30B_unique)
    print("qwen_30B_proportion: ", qwen_30B_proportion)

    ref_unique, qwen_480B_unique, qwen_480B_proportion = diff_and_proportion(qwen_480B, ref)
    print("ref_unique: ", ref_unique)
    print("qwen_480B_unique: ", qwen_480B_unique)
    print("qwen_480B_proportion: ", qwen_480B_proportion)

    ref_unique, xai_unique, xai_proportion = diff_and_proportion(xai, ref)
    print("ref_unique: ", ref_unique)
    print("xai_unique: ", xai_unique)
    print("xai_proportion: ", xai_proportion)

    plot_venn_simple(
        proportion=qwen_30B_proportion,
        ref_label="SweRank-32B",
        candidate_label="Qwen3-Coder-30B",
        title='SweRank vs Qwen-Coder-30B',
        savepath=os.path.join("plots", "qwen_30B_proportion")
    )

    plot_venn_simple(
        proportion=qwen_480B_proportion,
        ref_label="SweRank-32B",
        candidate_label="Qwen3-Coder-480B",
        title='(%): SweRank vs Qwen-Coder-480B',
        savepath=os.path.join("plots", "qwen_480B_proportion")
    )

    plot_venn_simple(
        proportion=xai_proportion,
        ref_label="SweRank-32B",
        candidate_label="Grok-Code-Fast-1",
        title='Remained Error Composition (%): SweRank vs Grok-Code-Fast-1',
        savepath=os.path.join("plots", "xai_proportion")
    )

    # create_academic_pie_chart(
    #     proportions=qwen_30B_proportion,
    #     labels=['SweRank', 'Common (IoU)', 'Qwen3-30B'],
    #     filename=os.path.join("plots", "qwen_30B_proportion"),
    #     title='Prediction Error Composition: SweRank vs Qwen-30B'
    # )
    # create_academic_pie_chart(
    #     proportions=qwen_480B_proportion,
    #     labels=['SweRank', 'Common (IoU)', 'Qwen3-480B'],
    #     filename=os.path.join("plots", "qwen_480B_proportion"),
    #     title='Prediction Error Composition: SweRank vs Qwen-480B'
    # )
    # create_academic_pie_chart(
    #     proportions=xai_proportion,
    #     labels=['SweRank', 'Common (IoU)', 'Grok-Code-Fast-1'],
    #     filename=os.path.join("plots", "xai_proportion"),
    #     title='Analysis of Set Overlap'
    # )

# 查看所有的36个内容，分析：19个公共问题，每个ref_unqiue（agent解决）的最大问题，每个candidate_unique (agent未解决)的最大问题
if __name__ == "__main__":
    main()