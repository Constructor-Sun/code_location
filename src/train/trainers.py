import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def graph_collate_fn(batch):
    queries, categories, graph_data_list = zip(*batch)
    
    batch_graph = Batch.from_data_list(graph_data_list)
    queries = torch.tensor(queries)
    categories = torch.tensor(categories)
    
    return queries, categories, batch_graph

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
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
        optimizer_name="Adam",
        lr=0.001,
        scheduler_name="ReduceLROnPlateau",
        scheduler_params=None,
        batch_size=16,
        epochs=10,
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

        # Move model to device(s)
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
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
        focal_loss_params = focal_loss_params or {'alpha': 0.25, 'gamma': 2.0}
        self.criterion = FocalLoss(**focal_loss_params)

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
        Train the GCNReasoner model and log loss and precision.
        
        Returns:
            List of average epoch losses
        """
        self.model.train()
        losses = []

        # Create dataset from data_list
        dataset = []
        for query, category, pt_path in zip(self.data_list['x'], self.data_list['y'], self.data_list['image']):
            graph_data = self.load_graph(os.path.join(self.graph_embedding, pt_path + '.pt'))
            dataset.append((query, category, graph_data))

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=graph_collate_fn)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            all_preds = []
            all_labels = []

            for batch in data_loader:
                self.optimizer.zero_grad()

                # Unpack batch
                query, category, graph_data = batch
                query = query.to(self.device)
                category = category.to(self.device)
                graph_data = graph_data.to(self.device)

                # Forward pass
                output = self.model(graph_data.x, graph_data.edge_index, query)
                loss = self.criterion(output, category)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Collect predictions and labels for precision
                pred = output.argmax(dim=1).cpu().numpy()
                labels = category.cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(labels)

                # Update scheduler if applicable
                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)

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

        # Save metrics to file
        self.save_metrics()

        return losses

    def evaluate(self, test_data_list):
        """
        Evaluate the model on test data.
        
        Args:
            test_data_list: List of tuples (query_vector, category_vector, graph_pt_path)
        
        Returns:
            float: Average test loss
            dict: Precision and recall for each class
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        dataset = []
        for query, category, pt_path in test_data_list:
            graph_data = self.load_graph(pt_path)
            dataset.append((query, category, graph_data))

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in data_loader:
                query, category, graph_data = batch
                query = query.to(self.device)
                category = category.to(self.device)
                graph_data = graph_data.to(self.device)

                output = self.model(graph_data.x, graph_data.edge_index, query)
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
            'model_state_dict': (self.model.module if isinstance(self.model, nn.DataParallel) else self.model).state_dict()
        }
        model_name = os.path.join(self.checkpoint_dir, f"{self.experiment_name}-{reason}.ckpt")
        torch.save(checkpoint, model_name)
        print(f"Best {reason}, save model as {model_name}")

    def load_checkpoint(self, filename):
        """
        Load a checkpoint from a file.
        
        Args:
            filename: Path to the checkpoint file
        """
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        
        # Handle DataParallel case
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(model_state_dict, strict=False)
        model.to(self.device)
        print(f"Loaded checkpoint from {filename}")