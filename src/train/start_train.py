import os
import argparse
import torch
from datasets import load_from_disk
from models import GCNReaonser
from trainers import GCNReasonerTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="train_index/SWE-smith/question_and_labels/train_dataset")
    parser.add_argument("--graph_embedding", type=str, default="train_index/SWE-smith/graph_embedding_pool")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--save_path", type=str, default="saved_models")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_processes', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    
    model = GCNReaonser(
        num_classes=2
    )

    trainer = GCNReasonerTrainer(
        model = model,
        data_list = dataset,
        graph_embedding = args.graph_embedding,
        embedding_model = args.embedding_model,
        save_path = args.save_path,
        log_file = "log_metrics.json",
        lr = args.lr,
        optimizer_name = args.optimizer,
        batch_size = args.batch_size,
        epochs = args.epochs
    )

    trainer.train()

if __name__ == "__main__":
    main()
