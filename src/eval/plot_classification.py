import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_academic_horizontal_barplot(raw_dict, save_path='academic_barplot.png'):
    data = {
        'Category': list(raw_dict.keys()),
        'Count': [len(v) for v in raw_dict.values()]
    }
    df = pd.DataFrame(data)

    df['Count'] = df['Count'].astype(int)
    total_count = df['Count'].sum()

    # 2. 图表设置
    sns.set_theme(style="whitegrid", context="talk")
    
    # 获取 Paired 颜色映射中的颜色
    colors = sns.color_palette("Paired", len(df))
    
    # 3. 绘图
    plt.figure(figsize=(10, 3))
    
    # 使用 seaborn.barplot 绘制横向柱状图
    ax = sns.barplot(
        x='Count', 
        y='Category', 
        data=df, 
        palette=colors,
        edgecolor='black',
        linewidth=1.0,
        width=0.5
    )

    sns.despine(left=True, bottom=True)
    for i, (count, category) in enumerate(zip(df['Count'], df['Category'])):
        percentage = (count / total_count) * 100
        # 修改这里：只显示百分比，不显示具体数量
        ax.text(
            count + 0.5,
            i, 
            f'{count} ({percentage:.1f}%)',
            va='center', 
            ha='left',
            fontsize=14,
            fontweight='bold'
        )

    # 设置轴标签和标题
    ax.set_title(
        'Reasons for LLM failure on Code Localization', 
        fontsize=18,
        pad=20,
        loc='center'
    )
    
    ax.set_xlabel('')
    ax.set_ylabel('') 
    
    ax.set_xlim(0, max(df['Count']) * 1.3) 

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")

def main():
    """主函数"""
    # 配置文件路径
    json_file_path = "datasets/classification.json"  # 请根据实际文件路径修改
    output_image_path = "plots/classification.png"  # 输出图片路径
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)
    create_academic_horizontal_barplot(raw_data, output_image_path)

if __name__ == "__main__":
    main()