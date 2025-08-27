import os
from typing import Optional
from datasets import load_dataset
from utils.benchmark.git_repo_manager import setup_github_repo
import argparse

import requests
import re

def get_image_metadata(image_name):
    """
    通过Docker Hub API获取镜像的元数据标签
    """
    # 提取镜像名部分（去掉组织名）
    image_only = image_name.split('/')[-1] if '/' in image_name else image_name
    
    # 使用Docker Hub API v2
    url = f"https://hub.docker.com/v2/repositories/jyangballin/{image_only}/tags/latest"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # 检查镜像的labels
        labels = data.get('images', [{}])[0].get('labels', {})
        return labels
        
    except Exception as e:
        print(f"获取镜像元数据失败: {e}")
        return {}

def find_base_commit(image_name):
    """
    主函数：从镜像名找到base_commit
    """
    # 1. 如果直接解析失败，尝试获取镜像元数据
    print("尝试通过Docker Hub API获取元数据...")
    metadata = get_image_metadata(image_name)
    
    # 检查常见的包含提交信息的label
    commit_labels = [
        'org.opencontainers.image.revision',
        'vcs-ref', 
        'commit',
        'git.commit',
        'source.commit'
    ]
    
    for label in commit_labels:
        if label in metadata:
            print(f"从label '{label}' 找到提交: {metadata[label]}")
            # 还需要确定仓库名，可能从其他label获取
            repo_label = None
            for repo_label_name in ['org.opencontainers.image.source', 'vcs-url']:
                if repo_label_name in metadata:
                    repo_label = metadata[repo_label_name]
                    break
            
            if repo_label and 'github.com' in repo_label:
                # 从URL中提取仓库路径
                repo_path = repo_label.split('github.com/')[-1].replace('.git', '')
                return repo_path, metadata[label]
    
    # 3. 如果以上都失败，抛出错误
    raise ValueError(f"无法从镜像 {image_name} 中确定仓库和提交信息")

def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}


def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]


def setup_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id, dataset, split)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")
    
    if dataset == "princeton-nlp/SWE-bench_Lite" and split == "test":
        repo_dir_name = instance_data["repo"].replace("/", "__")
        github_repo_path = f"swe-bench/{repo_dir_name}"
    else:
        github_repo_path = instance_data["repo"]
    base_commit = instance_data.get("base_commit", None)
    if base_commit is None:
        _, base_commit = find_base_commit(instance_data["image_name"])
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=base_commit,
        base_dir=repo_base_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repo_base_dir", type=str, default='/tmp/repos')
    parser.add_argument("--eval_n_limit", type=int, default=1)

    args = parser.parse_args()

    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    if args.eval_n_limit:
        swe_bench_data = swe_bench_data.select(range(args.eval_n_limit))
    
    for instance in swe_bench_data:
        # repo_base_dir = os.path.join(args.repo_base_dir, instance['instance_id'])
        path = setup_repo(instance_data=instance, repo_base_dir=args.repo_base_dir)
        print(instance['instance_id'], path)