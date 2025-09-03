import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import DataParallel # , DistributedDataParallel
from torch_geometric.loader import DataListLoader
from dataset import GraphDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

# def graph_collate_fn(batch):
#     queries, categories, graph_data_list = zip(*batch)
    
#     batch_graph = Batch.from_data_list(graph_data_list)
#     queries = torch.tensor(queries)
#     categories = torch.tensor(categories)
    
#     return queries, categories, batch_graph

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        """
        Focal Loss for imbalanced datasets.
        
        Args:
            alpha: Weighting factor for the positive class (default: 0.25 for imbalanced datasets)
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean' or 'sum' for loss reduction
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class GCNReasonerTrainer:
    def __init__(
        self,
        model,
        data_list,
        graph_embedding,
        num_classes=2,
        optimizer_name="Adam",
        lr=0.001,
        scheduler_name="ReduceLROnPlateau",
        scheduler_params=None,
        batch_size=16,
        epochs=10,
        log_file=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        focal_loss_params=None
    ):
        """
        Trainer for GCNReasoner model with multi-GPU support and focal loss.
        
        Args:
            model: GCNReasoner instance
            data_list: List of tuples (query_vector, category_vector, graph_pt_path)
            optimizer_name: String, e.g., "Adam", "SGD"
            lr: Learning rate
            scheduler_name: String, e.g., "ReduceLROnPlateau", "StepLR"
            scheduler_params: Dict of scheduler parameters
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to train on ("cuda" or "cpu")
            focal_loss_params: Dict with 'alpha' and 'gamma' for focal loss
        """
        self.model = model
        self.data_list = data_list
        self.graph_embedding = graph_embedding
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.log_file = log_file
        self.num_classes = num_classes

        # Move model to device(s)
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(self.model)
        self.model = self.model.to(device)

        # Initialize optimizer
        optimizer_class = getattr(optim, optimizer_name, optim.Adam)
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)

        # Initialize scheduler
        scheduler_params = scheduler_params or {}
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_name == "StepLR":
            self.scheduler = StepLR(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None

        # Initialize focal loss
        # focal_loss_params = focal_loss_params or {'alpha': 0.25, 'gamma': 2.0}
        # self.criterion = FocalLoss(**focal_loss_params)
        self.criterion = nn.KLDivLoss(reduction='batchmean')

        # Initialize metrics storage
        self.training_metrics = []

    def log_metrics(self, epoch, avg_loss, precision_0, precision_1):
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
            precision_0: Precision for class 0
            precision_1: Precision for class 1
        """
        metrics = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'precision_class_0': precision_0,
            'precision_class_1': precision_1
        }
        self.training_metrics.append(metrics)
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
              f"Precision (Class 0): {precision_0:.4f}, Precision (Class 1): {precision_1:.4f}")

    def save_metrics(self):
        """
        Save training metrics to a JSON file.
        """
        with open(self.log_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)

    def load_graph(self, pt_path):
        """
        Load graph data from .pt file.
        
        Args:
            pt_path: Path to .pt file containing graph data
        
        Returns:
            Data: Torch geometric Data object
        """
        return torch.load(pt_path, weights_only=False)

    def train(self):
        """
        Train the GCNReasoner model, log loss and precision, and save the model.
        
        Returns:
            List of average epoch losses
        """
        self.model.train()
        losses = []

        # Create custom dataset for lazy loading
        dataset = GraphDataset(
            queries=self.data_list['x'],
            categories=self.data_list['y'],
            pt_paths=self.data_list['image'],
            graph_embedding=self.graph_embedding
        )

        # Use torch_geometric.loader.DataLoader
        data_loader = DataListLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            all_preds = []
            all_labels = []
            batch_pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                         leave=False, unit="batch")

            for batch_idx, batchs in enumerate(batch_pbar):
                self.optimizer.zero_grad()

                # Move batch to device
                # Extract query, category, and graph data
                category = [data.category for data in batchs]
                ori_category = torch.cat(category)
                category = F.one_hot(ori_category, num_classes=self.num_classes).cuda()
                category = category / category.sum(dim=1, keepdim=True)

                # Forward pass
                output = self.model(batchs)
                output = F.log_softmax(output, dim=-1)
                loss = self.criterion(output, category)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                current_loss = loss.item()
                batch_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })

                # Collect predictions and labels for precision
                pred = output.argmax(dim=1).cpu().numpy()
                labels = ori_category.cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(labels)

                # Update scheduler if applicable
                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss.item())

                torch.cuda.empty_cache()

            # Average loss for the epoch
            avg_loss = epoch_loss / len(data_loader)
            losses.append(avg_loss)

            # Calculate precision for each class
            precision, _, _, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=[0, 1], zero_division=0
            )

            # Log metrics
            self.log_metrics(epoch, avg_loss, precision[0], precision[1])

            # Step scheduler if not ReduceLROnPlateau
            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            # Save metrics and model
            self.save_metrics()
            self.save_checkpoint()

        return losses

    def evaluate(self, test_data_list):
        """
        Evaluate the model on test data.
        
        Args:
            test_data_list: Dict with keys 'x' (queries), 'y' (categories), 'image' (pt_paths)
        
        Returns:
            float: Average test loss
            dict: Precision and recall for each class
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Create custom dataset for lazy loading
        dataset = GraphDataset(
            queries=test_data_list['x'],
            categories=test_data_list['y'],
            pt_paths=test_data_list['image'],
            graph_embedding=self.graph_embedding
        )

        # Use torch_geometric.loader.DataLoader
        data_loader = DataListLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(self.device)

                # Extract query, category, and graph data
                query = batch.query
                category = batch.category
                x = batch.x
                edge_index = batch.edge_index

                # Forward pass
                output = self.model(x, edge_index, query)
                loss = self.criterion(output, category)
                total_loss += loss.item()

                # Collect predictions and labels for metrics
                pred = output.argmax(dim=1).cpu().numpy()
                labels = category.cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(labels)

        avg_loss = total_loss / len(data_loader)

        # Calculate precision and recall for each class
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1], zero_division=0
        )
        metrics = {
            'precision_class_0': precision[0],
            'precision_class_1': precision[1],
            'recall_class_0': recall[0],
            'recall_class_1': recall[1]
        }

        return avg_loss, metrics

    def save_checkpoint(self, reason="h1"):
        """
        Save a checkpoint of the model.
        
        Args:
            reason: String identifier for the checkpoint (default: "h1")
        """
        checkpoint = {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epoch': self.epochs
        }
        model_name = os.path.join(self.checkpoint_dir, f"{self.experiment_name}-{reason}.ckpt")
        torch.save(checkpoint, model_name)
        print(f"Best {reason}, save model as {model_name}")

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """
        Load a checkpoint and create a Trainer instance.
        
        Args:
            path (str): File path to load the checkpoint from (e.g., 'checkpoint.pth')
            device (str): Device to load the model to ('cpu' or 'cuda')
        
        Returns:
            Trainer: Loaded Trainer instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        trainer = cls(model, optimizer, device)
        trainer.epoch = checkpoint['epoch']
        trainer.model.eval()
        print(f"Checkpoint loaded from {path}, epoch {trainer.epoch}")
        return trainer