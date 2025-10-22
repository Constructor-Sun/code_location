import json
import matplotlib.pyplot as plt
from upsetplot import from_contents, plot
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_upset_plot_from_json(json_file_path, save_path, target_size=19):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    print("current results: ", data_dict.keys())

    upset_data = from_contents(data_dict)
    print("upset_data: ", upset_data)
    print("upset_data: ", type(upset_data))
    upset_data.name = "(Count of Elements)" 
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    plot(
        upset_data, 
        fig=plt.gcf(), 
        orientation='horizontal',
        sort_by='cardinality',
        show_counts=True
    )
    
    plt.suptitle(f'Failure Instance Overlap Across Models', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def main():
    file_path = "datasets/swe-bench-lite-subset-summary.json"
    save_path = "plots/upset.png"
    create_upset_plot_from_json(file_path, save_path)

if __name__ == "__main__":
    main()