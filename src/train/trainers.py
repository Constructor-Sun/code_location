import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
# torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100000, trace_alloc_record_context=True)
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import DataParallel # , DistributedDataParallel
from torch_geometric.loader import DataListLoader
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import subgraph
from dataset import GraphDataset
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from collections import deque

# def graph_collate_fn(batch):
#     queries, categories, graph_data_list = zip(*batch)
    
#     batch_graph = Batch.from_data_list(graph_data_list)
#     queries = torch.tensor(queries)
#     categories = torch.tensor(categories)
    
#     return queries, categories, batch_graph
VERY_SMALL_NUMBER = 1e-10

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
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embed_max_length=2048,
        num_classes=2,
        optimizer_name="Adam",
        lr=0.001,
        scheduler_name="ReduceLROnPlateau",
        scheduler_params=None,
        batch_size=16,
        epochs=10,
        save_path="saved_models",
        log_file=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.5, 
        max_nodes=10,
        top_k=5,
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
        self.embedding_model = embedding_model
        self.embed_max_length = embed_max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = save_path
        self.log_file = os.path.join(save_path, log_file)
        self.threshold = threshold
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.num_neighbors = [40, 40, 30]
        self.subgraph_limit = np.prod(self.num_neighbors) / 3 * 2

        # Move model to device(s)
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = DataParallel(self.model)
        self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, trust_remote_code=True)
        self.embed_model = AutoModel.from_pretrained(self.embedding_model, trust_remote_code=True)
        self.embed_model = self.embed_model.to(device)

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
        self.criterion = nn.KLDivLoss(reduction='sum')

        # Initialize metrics storage
        self.training_metrics = []

    def log_metrics(self, epoch, avg_loss, precision, recall):
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
            precision: 
            recall: 
        """
        metrics = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'precision': precision,
            'recall': recall
        }
        self.training_metrics.append(metrics)
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, "
              f"Precision (Class 0): {precision:.4f}, Precision (Class 1): {recall:.4f}")

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
    

    # def kl_loss(self, pred_list, target_list):
    #     """
    #     Calculate KL divergence loss for a batch of predictions and targets.
        
    #     Args:
    #         pred_list (list): List of tensors, each of shape [num_nodes, 2], containing logits.
    #         target_list (list): List of tensors, each of shape [num_nodes], containing binary labels (0 or 1).
            
    #     Returns:
    #         torch.Tensor: Average KL divergence loss across the batch.
    #     """
    #     assert len(pred_list) == len(target_list), "pred_list and target_list must have the same length"
    #     batch_size = len(pred_list)
    #     total_loss = 0.0
    #     valid_samples = 0
        
    #     for i in range(batch_size):
    #         pred = pred_list[i]  # Shape: [num_nodes, 2]
    #         target = target_list[i]  # Shape: [num_nodes]
            
    #         # Convert logits to probabilities via softmax
    #         pred_prob = F.softmax(pred, dim=1)  # Shape: [num_nodes, 2]
    #         pred_prob = pred_prob.clamp(min=VERY_SMALL_NUMBER)  # Avoid log(0)
            
    #         # Convert target to probability distribution
    #         target_prob = torch.zeros_like(pred)  # Shape: [num_nodes, 2]
    #         target_prob[:, 1] = target.float()  # Positive class (1) probability
    #         target_prob[:, 0] = 1.0 - target.float()  # Negative class (0) probability
            
    #         # Normalize target to sum to 1 per node (if not already normalized)
    #         answer_number = torch.sum(target)
    #         if answer_number > 0:  # Only include samples with at least one positive label
    #             # Compute KL divergence
    #             log_pred_prob = torch.log(pred_prob)
    #             loss = self.criterion(log_pred_prob, target_prob)  # Shape: scalar (sum reduction)
    #             total_loss += loss
    #             valid_samples += 1
        
    #     # Compute average loss
    #     if valid_samples == 0:
    #         return torch.tensor(0.0, device=pred_list[0].device, requires_grad=True)
    #     avg_loss = total_loss / valid_samples
    #     return avg_loss

    def kl_loss(self, pred_list, target_list):
        assert len(pred_list) == len(target_list), "pred_list and target_list must have the same length"
        batch_size = len(pred_list)
        total_loss = 0.0
        valid_samples = 0

        for i in range(batch_size):
            pred = pred_list[i]
            target = torch.clamp(target_list[i], min=0.0)
            target_sum = target.sum()
            if target_sum > 1e-8:
                target = target / target.sum()
                pred_log = torch.log(torch.clamp(pred, min=1e-8))
                loss = self.criterion(pred_log, target)  # Shape: scalar (sum reduction)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss
                    valid_samples += 1
                else:
                    print('pred_list in non loss: ', pred_list)
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=pred_list[0].device, requires_grad=True) 
        
        avg_loss = total_loss / valid_samples
        return avg_loss
    
    def get_pool_query(self, query, query_mask):
        masked_query = query * query_mask
    
        valid_token_count = query_mask.sum(dim=1)  # [batch_query_num, 1]
        valid_token_count = torch.clamp(valid_token_count, min=1.0)
        
        # pooling at dim 1
        sum_query = masked_query.sum(dim=1)  # [batch_query_num, embed_dim]
        mean_query = sum_query / valid_token_count
        return mean_query
    
    def init_nodes(self, x, query_pool, batch, top_k=5):
        """
        batchs x: node embeddings [batch_nodes_num, embed_dim]
        batchs query_pool: query [batch_query_num, embed_dim]
        batchs batch: node embedding index in current batch [batch_nodes_num]
        top_k: select top_k nodes as initial nodes
        """
        x = x.to(query_pool.device)
        batch_size = query_pool.shape[0]
        num_nodes = x.shape[0]
        
        # expand query to each graph, then compute cosine similarity
        query_expanded = query_pool[batch]  # [batch_nodes_num, embed_dim]
        similarities = F.cosine_similarity(x, query_expanded, dim=1)  # [batch_nodes_num]

        # mask
        p = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        for i in range(batch_size):
            # i-th graph indices
            graph_indices = (batch == i).nonzero(as_tuple=True)[0]
            
            if len(graph_indices) == 0:
                continue
                
            graph_similarities = similarities[graph_indices]
            _, topk_local_indices = torch.topk(
                graph_similarities, 
                k=min(top_k, len(graph_indices))
            )
            
            # top_k indices in one graph
            topk_global_indices = graph_indices[topk_local_indices]
            p[topk_global_indices] = 1.0
        
        return p / top_k
    
    def get_min_distance_to_target_optimized(self, data):
        seed_nodes = data.p_0.cpu().numpy() if torch.is_tensor(data.p_0) else np.array(data.p_0)
        target_nodes = torch.where(data.category == 1)[0].cpu().numpy()
        
        if len(seed_nodes) == 0 or len(target_nodes) == 0:
            return -1
        
        # 构建邻接表
        edge_index = data.edge_index.cpu().numpy()
        adj_list = {}
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_list.setdefault(src, []).append(dst)
            adj_list.setdefault(dst, []).append(src)
        
        # 多源BFS从种子节点开始
        visited_from_seed = {}
        queue_from_seed = deque()
        
        for seed in seed_nodes:
            visited_from_seed[seed] = 0
            queue_from_seed.append(seed)
        
        # 从目标节点开始的BFS
        visited_from_target = {}
        queue_from_target = deque()
        
        for target in target_nodes:
            visited_from_target[target] = 0
            queue_from_target.append(target)
        
        # 双向BFS
        while queue_from_seed and queue_from_target:
            # 从种子节点方向扩展
            current_size = len(queue_from_seed)
            for _ in range(current_size):
                node = queue_from_seed.popleft()
                
                if node in visited_from_target:
                    return visited_from_seed[node] + visited_from_target[node] - 1
                
                if node in adj_list:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited_from_seed:
                            visited_from_seed[neighbor] = visited_from_seed[node] + 1
                            queue_from_seed.append(neighbor)
            
            # 从目标节点方向扩展
            current_size = len(queue_from_target)
            for _ in range(current_size):
                node = queue_from_target.popleft()
                
                if node in visited_from_seed:
                    return visited_from_seed[node] + visited_from_target[node] - 1
                
                if node in adj_list:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited_from_target:
                            visited_from_target[neighbor] = visited_from_target[node] + 1
                            queue_from_target.append(neighbor)
        
        return -1

    def train(self):
        """
        Train the GCNReasoner model, log loss and precision, and save the model.
        
        Returns:
            List of average epoch losses
        """
        self.model.train()
        self.embed_model.eval()
        losses = []

        # Create custom dataset for lazy loading
        dataset = GraphDataset(
            queries=self.data_list['x'],
            # queries_mask=self.data_list['x_mask'],
            categories=self.data_list['y'],
            pt_paths=self.data_list['image'],
            graph_embedding=self.graph_embedding
        )

        # Use torch_geometric.loader.DataLoader
        data_loader = DataListLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            all_predictions = []
            all_labels = []
            batch_pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                         leave=False, unit="batch")

            for batch_idx, batchs in enumerate(batch_pbar):
                self.optimizer.zero_grad()
                # Move batch to device
                # Extract query, category, and graph data
                query = []
                category = []
                for data in batchs:
                    query.append(data.query)

                # get issue(query) embeddings
                inputs = self.tokenizer(query, return_tensors="pt", truncation=True, 
                       max_length=self.embed_max_length, padding="longest")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    query_embeddings = self.embed_model(**inputs).last_hidden_state
                query_pool = self.get_pool_query(query_embeddings, inputs["attention_mask"].unsqueeze(-1))
                for i, data in enumerate(batchs):
                    data.query = query_embeddings[i].unsqueeze(0).cpu()
                    data.query_mask = inputs["attention_mask"][i].view(1, -1, 1).cpu()
                    data.query_pool = query_pool[i].unsqueeze(0).cpu()
                    data.p_0 = self.init_nodes(data.x, 
                                               query_pool[i].unsqueeze(0), 
                                               torch.zeros(data.x.size(0), dtype=torch.int).to(self.device), 
                                               top_k=self.top_k).cpu()
                    # distance = self.get_min_distance_to_target_optimized(data)
                    # print("distance: ", distance)
                    if data.x.shape[0] > self.subgraph_limit:
                        # print("before subgraph: ", data.x.shape[0])
                        target_nodes = torch.where(data.category == 1.)[0]
                        seed_nodes = torch.where(data.p_0 > 0)[0]
                        node_idx = torch.cat([seed_nodes, target_nodes]).unique()

                        sampler = NeighborSampler(data.edge_index, node_idx=node_idx,
                                sizes=self.num_neighbors, batch_size=len(node_idx),
                                shuffle=False, num_workers=4)
                        
                        _, n_id, _ = next(iter(sampler))
                        subgraph_data = data.subgraph(n_id)
                        data.x = subgraph_data.x
                        data.edge_index = subgraph_data.edge_index
                        data.p_0 = data.p_0[n_id]
                        data.category = data.category[n_id]
                    category.append(data.category)
    
                # Forward pass
                # reporter = MemReporter(self.model)
                logit = self.model(batchs)
                logits = []
                begin = 0
                for label in category:
                    pred = logit[begin:begin+len(label)]
                    begin += len(label)
                    logits.append(pred)
                category = [label.to(self.device) for label in category]
                loss = self.kl_loss(logits, category)

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
                predictions = self.get_predict_nodes(logits, self.threshold, self.max_nodes)
                predictions = [prediction.cpu() for prediction in predictions]
                category = [cate.cpu() for cate in category]
                all_predictions.extend(predictions)
                all_labels.extend(category)

                # Update scheduler if applicable
                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss.item())

                torch.cuda.empty_cache()
                gc.collect()

            # Average loss for the epoch
            avg_loss = epoch_loss / len(data_loader)
            losses.append(avg_loss)

            # Calculate precision for each class
            precision, recall = self.precision_and_recall(
                all_labels, all_predictions
            )

            # Log metrics
            self.log_metrics(epoch, avg_loss, precision, recall)

            # Step scheduler if not ReduceLROnPlateau
            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            # Save metrics and model
            self.save_metrics()
            self.save_checkpoint()

        return losses
    
    def get_predict_nodes(self, logits, threshold = 0.5, max_nodes = 10):
        predictions = []
        
        for i, logit in enumerate(logits):
            probs = logit
            # if logit.dim() == 1:
            #     probs = logit
            # else:
            #     probs = torch.softmax(logit, dim=-1)
            
            # sort in a descendent way
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            selected_indices = []
            cumulative_prob = 0.0
            count = 0
            
            # select nodes
            for j in range(len(sorted_probs)):
                if (cumulative_prob + sorted_probs[j].item() <= threshold and 
                    count < max_nodes and 
                    sorted_probs[j].item() > 1e-8):  # 忽略概率极小的节点
                    
                    selected_indices.append(sorted_indices[j].item())
                    cumulative_prob += sorted_probs[j].item()
                    count += 1
                else:
                    break
            
            predictions.append(torch.tensor(selected_indices, dtype=torch.long))
        
        return predictions

    def precision_and_recall(self, predictions, labels):
        assert len(predictions) == len(labels), "Predictions and labels must have the same length"
        
        total_precision = 0.0
        total_recall = 0.0
        total_hits = 0.0
        num_samples = len(predictions)
        
        for pred, label in zip(predictions, labels):
            pred_set = set(pred.tolist())
            label_set = set(label.tolist())
            
            tp = len(pred_set & label_set)
            fp = len(pred_set - label_set)
            fn = len(label_set - pred_set)
            
            # precision, recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1-score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            # hits
            hits = 1.0 if len(pred_set & label_set) > 0 else 0.0
            
            total_precision += precision
            total_recall += recall
            total_hits += hits
        
        avg_precision = total_precision / num_samples
        avg_recall = total_recall / num_samples
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        avg_hits = total_hits / num_samples
        
        return avg_precision, avg_recall

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
        all_predictions = []
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
                all_predictions.extend(pred)
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
        model_name = os.path.join(self.checkpoint_dir, f"CodeReasoner-{reason}.ckpt")
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