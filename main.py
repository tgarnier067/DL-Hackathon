import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Assume GNN and GraphDataset are imported/defined elsewhere
from source.models import GNN
from source.loadData import GraphDataset

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

class WCELoss(torch.nn.Module):
    def __init__(self, gamma=0.2):
        super(WCELoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        probs = F.softmax(logits, dim=1)
        true_probs = probs[range(len(labels)), labels]
        weight = (true_probs.detach() ** self.gamma)
        loss = weight * ce_loss
        return loss.mean()

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct / total

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    os.makedirs(submission_folder, exist_ok=True)
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="Train or test the GNN model.")
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test.json.gz file')
    parser.add_argument('--train_path', type=str, default=None, help='Path to the train.json.gz file (optional)')

    args = parser.parse_args()

    test_path = args.test_path
    train_path = args.train_path

    # Parameters
    num_checkpoints = 20
    gnn = 'gcn-virtual'
    drop_ratio = 0.5
    num_layer = 5
    emb_dim = 300
    batch_size = 32
    epochs = 200
    baseline_mode = 2
    num_epochs = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if train_path:
        print(f"Training mode: training with {train_path} and testing with {test_path}")
    
        test_dir_name = os.path.basename(os.path.dirname(args.test_path)) # A, B, C, D
        script_dir = os.getcwd()
        logs_folder = os.path.join(script_dir, "logs", test_dir_name)
        os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "training.log")
    
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    
        checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
        checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
        os.makedirs(checkpoints_folder, exist_ok=True)
    
        # Modelization
        model = GNN(gnn_type='gcn', num_class=6, num_layer=num_layer, emb_dim=emb_dim, drop_ratio=drop_ratio, virtual_node=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = WCELoss(gamma=0.2)


        
        # Training phase
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
    
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
            
        best_val_accuracy = 0.0
    
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
    
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]
    
        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
    
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")
    
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))


        
        import gc
        del train_dataset
        del train_loader
        del full_dataset
        del val_dataset
        del val_loader
        gc.collect()

        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.load_state_dict(torch.load(checkpoint_path))
        predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
        save_predictions(predictions, args.test_path)


        
    else:
        print(f"Inference mode: testing with {test_path} only")
        # Appelle ta fonction d'inférence ici

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GNN(gnn_type='gcn', num_class=6, num_layer=num_layer,
            emb_dim=emb_dim, drop_ratio=drop_ratio,
            virtual_node=True).to(device)

        test_dataset = GraphDataset(args.test_path, transform=add_zeros)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_dir_name = os.path.basename(os.path.dirname(args.test_path)) # A, B, C, D
        script_dir = os.getcwd()
        logs_folder = os.path.join(script_dir, "logs", test_dir_name)
        os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, "training.log")
    
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    
        checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
        checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
        os.makedirs(checkpoints_folder, exist_ok=True)

        # Load pre-trained model if no training path given
        if os.path.exists(checkpoint_path) and not args.train_path:
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded best model from {checkpoint_path}")
            predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
            save_predictions(predictions, args.test_path)
            

if __name__ == "__main__":
    main()

    
