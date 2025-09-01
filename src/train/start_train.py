import argparse
import torch
from datasets import load_from_disk
from models import GCNReaonser
from trainers import GCNReasonerTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="train_index/SWE-smith/question_and_labels/train_dataset")
    parser.add_argument("--graph_embedding", type=str, default="train_index/SWE-smith/graph_embedding")
    parser.add_argument("--save_path", type=str, default="saved_models")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_processes', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    in_channels = len(dataset[0]['x'])
    
    model = GCNReaonser(
        in_channels = in_channels,
        hidden_channels = in_channels,
        num_classes=2
    )

    trainer = GCNReasonerTrainer(
        model = model,
        data_list = dataset,
        graph_embedding = args.graph_embedding,
        lr = args.lr,
        optimizer_name = args.optimizer,
        batch_size = args.batch_size,
        epochs = args.epochs
    )

    trainer.train()

if __name__ == "__main__":
    main()
