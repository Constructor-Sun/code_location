import pickle
import networkx as nx

def read_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # 打印数据类型
        print(f"数据类型: {type(data)}")
        
        # 检查是否为 NetworkX 的 MultiDiGraph
        if isinstance(data, nx.MultiDiGraph):
            print("\n这是一个 NetworkX MultiDiGraph，内容概览：")
            print(f"节点数: {data.number_of_nodes()}")
            print(f"边数: {data.number_of_edges()}")
            
            # 打印前几个节点（最多 5 个）
            print("\n前几个节点（最多 5 个）：")
            nodes = list(data.nodes(data=True))[:5]  # 获取节点及其属性
            for node, attrs in nodes:
                print(f"节点: {node}, 属性: {attrs}")
            
            # 打印前几条边（最多 5 条）
            print("\n前几条边（最多 5 条）：")
            edges = list(data.edges(data=True))[:5]  # 获取边及其属性
            for u, v, attrs in edges:
                print(f"边: ({u} -> {v}), 属性: {attrs}")
        
        else:
            print("\n其他类型的数据，内容：")
            print(data)
        
        return data
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 替换为你的文件路径
file_path = 'train_index/SWE-smith/graph_index_v1.0/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9.pkl'  # 根据你的文件路径调整
data = read_pkl_file(file_path)