"""
train_tumor_model.py  (GPU‑aware, AMP‑optional)
------------------------------------------------
* Accepts --data_path that may be root or any split folder.
* Prints device info.
* Default batch_size 32 (fits 8 GB @ 384×384).
* Optional --amp to enable mixed‑precision.
* Added visualization for test results with prediction confidence.
* Saves confusion matrix and misclassified examples.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ---------- Path helper ---------------------------------------------------- #
def resolve_dataset_root(path: Path) -> Path:
    return path.parent if path.name.lower() in {"train", "valid", "test"} else path


# ---------- Dataset -------------------------------------------------------- #
class TumorDataset(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.folder = folder
        self.df = pd.read_csv(self.folder / "_classes.csv")
        self.image_paths = [self.folder / f for f in self.df.iloc[:, 0]]
        self.labels = self.df.iloc[:, 1:].fillna(0).values.astype(np.float32)
        self.transform = transform
        self.class_names = self.df.columns[1:].tolist()

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32), str(img_path)


# ---------- Utility -------------------------------------------------------- #
def pos_weights(labels):
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


class EarlyStop:
    def __init__(self, patience=5):
        self.best, self.wait, self.patience = 1e9, 0, patience

    def step(self, loss):
        improved = loss < self.best
        self.best = min(self.best, loss)
        self.wait = 0 if improved else self.wait + 1
        return self.wait >= self.patience


# ---------- Visualization functions ---------------------------------------- #
def visualize_predictions(images, true_labels, preds, image_paths, class_names, output_dir, threshold=0.5, max_images=20):
    """
    Visualizes model predictions alongside ground truth labels
    """
    os.makedirs(output_dir / "predictions", exist_ok=True)
    
    # Determine the subset of images to visualize
    total_imgs = len(images)
    indices = np.random.choice(total_imgs, min(max_images, total_imgs), replace=False)
    
    for i, idx in enumerate(indices):
        img = images[idx].permute(1, 2, 0)  # CHW -> HWC
        img = img * 0.5 + 0.5  # Denormalize: ([0,1] - 0.5) / 0.5 -> [0,1]
        
        # Get binary predictions
        pred_binary = preds[idx] > threshold
        true_binary = true_labels[idx] > 0.5
        
        # Create a figure with the image and predictions
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 1, 1)
        plt.imshow(img.numpy())
        
        # Get image filename from path
        img_name = Path(image_paths[idx]).name
        plt.title(f"Image: {img_name}")
        
        # Create text for predictions
        pred_text = ""
        for j, class_name in enumerate(class_names):
            true_val = true_binary[j].item()
            pred_val = pred_binary[j].item()
            pred_prob = preds[idx][j].item()
            
            color = "green" if true_val == pred_val else "red"
            pred_text += f"{class_name}: True={true_val}, Pred={pred_val} ({pred_prob:.2f})\n"
        
        plt.figtext(0.5, 0.01, pred_text, ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5}, color="black")
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / "predictions" / f"pred_{i}_{img_name}")
        plt.close()


def plot_confusion_matrices(true_labels, preds, class_names, output_dir, threshold=0.5):
    """
    Plots confusion matrices for each class
    """
    os.makedirs(output_dir / "confusion_matrices", exist_ok=True)
    
    pred_binary = (preds > threshold).numpy().astype(int)
    true_binary = true_labels.numpy().astype(int)
    
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(true_binary[:, i], pred_binary[:, i])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices" / f"cm_{class_name.replace(' ', '_')}.png")
        plt.close()


def save_misclassified(images, true_labels, preds, image_paths, class_names, output_dir, threshold=0.5, max_per_class=10):
    """
    Saves examples of misclassified images for each class
    """
    os.makedirs(output_dir / "misclassified", exist_ok=True)
    
    pred_binary = (preds > threshold).numpy().astype(int)
    true_binary = true_labels.numpy().astype(int)
    
    for class_idx, class_name in enumerate(class_names):
        # Find misclassified examples for this class
        misclassified = np.where(pred_binary[:, class_idx] != true_binary[:, class_idx])[0]
        
        if len(misclassified) == 0:
            continue
            
        # Select a subset of misclassified examples
        selected = np.random.choice(misclassified, min(max_per_class, len(misclassified)), replace=False)
        
        for i, idx in enumerate(selected):
            img = images[idx].permute(1, 2, 0)
            img = img * 0.5 + 0.5  # Denormalize
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img.numpy())
            
            img_name = Path(image_paths[idx]).name
            true_val = true_binary[idx, class_idx]
            pred_val = pred_binary[idx, class_idx]
            pred_prob = preds[idx, class_idx].item()
            
            plt.title(f"{class_name}: True={true_val}, Pred={pred_val} ({pred_prob:.2f})\n{img_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "misclassified" / f"{class_name.replace(' ', '_')}_{i}_{img_name}")
            plt.close()


# Function to run inference on a single image
def predict_single_image(model, image_path, transform, class_names, device, threshold=0.5):
    """
    Makes prediction on a single image and returns results
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = model(img_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
    
    # Create a results dictionary
    results = {
        'image_path': str(image_path),
        'predictions': []
    }
    
    for i, class_name in enumerate(class_names):
        results['predictions'].append({
            'class': class_name,
            'probability': float(probs[i]),
            'prediction': 'Positive' if probs[i] > threshold else 'Negative'
        })
    
    return results


# ---------- Training ------------------------------------------------------- #
def train(args):
    root = resolve_dataset_root(Path(args.data_path))
    train_dir, val_dir, test_dir = root / "train", root / "valid", root / "test"

    # -- transforms
    tf_train = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # -- datasets & loaders
    ds_tr, ds_va, ds_te = (TumorDataset(train_dir, tf_train),
                           TumorDataset(val_dir, tf_val),
                           TumorDataset(test_dir, tf_val))

    pin_mem = torch.cuda.is_available()
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,
                       pin_memory=pin_mem, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       pin_memory=pin_mem, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False,
                       pin_memory=pin_mem, num_workers=2)

    # -- device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟  Using device: {device}")
    if device.type == "cuda":
        print("🖥️   GPU:", torch.cuda.get_device_name(0))

    # -- model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, ds_tr.labels.shape[1])
    model.to(device)

    # -- loss, optim, sched
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(ds_tr.labels).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=3)
    stopper = EarlyStop(args.pat)

    # -- AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1 = 0
    args.out.mkdir(parents=True, exist_ok=True)

    # -------- epoch loop -------- #
    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train(); tr_loss = 0
        for x, y, _ in tqdm(dl_tr, desc=f"Epoch {ep} Train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(dl_tr.dataset)

        # ---- validate ----
        model.eval(); va_loss = 0; preds, labs = [], []
        with torch.no_grad():
            for x, y, _ in tqdm(dl_va, desc=f"Epoch {ep} Val"):
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = model(x)
                    loss = criterion(out, y)
                va_loss += loss.item() * x.size(0)
                preds.append(torch.sigmoid(out).cpu())
                labs.append(y.cpu())
        va_loss /= len(dl_va.dataset)
        preds_all, labs_all = torch.cat(preds), torch.cat(labs)
        f1 = f1_score(labs_all, (preds_all > 0.5), average="macro")

        print(f"Epoch {ep:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out / "best.pth")
        if stopper.step(va_loss):
            print("⏹️  Early stop.")
            break
        sched.step(va_loss)

    # ---- test ----
    model.load_state_dict(torch.load(args.out / "best.pth"))
    model.eval(); preds, labs, test_images, image_paths = [], [], [], []
    with torch.no_grad():
        for x, y, paths in tqdm(dl_te, desc="Testing"):
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds.append(torch.sigmoid(model(x)).cpu())
            labs.append(y)
            test_images.append(x.cpu())
            image_paths.extend(paths)
    
    preds_all = torch.cat(preds)
    labs_all = torch.cat(labs)
    test_images_all = torch.cat(test_images)
    
    # Calculate metrics
    print("\n🏁  Test F1:", f1_score(labs_all, (preds_all > 0.5), average="macro"))
    print("\n📄  Classification report:\n",
          classification_report(labs_all, (preds_all > 0.5), zero_division=0))
    
    # Visualize predictions
    if args.visualize:
        print("\n🔍 Generating visualizations...")
        visualize_predictions(test_images_all, labs_all, preds_all, image_paths, 
                             ds_te.class_names, args.out, max_images=args.num_vis)
        
        plot_confusion_matrices(labs_all, preds_all, ds_te.class_names, args.out)
        
        save_misclassified(test_images_all, labs_all, preds_all, image_paths,
                          ds_te.class_names, args.out)
        
        print(f"✅ Visualizations saved to {args.out}")

    # Save the model for inference
    inference_path = args.out / "model_inference.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': ds_te.class_names,
        'img_size': args.img
    }, inference_path)
    print(f"✅ Model saved for inference at {inference_path}")


# ---------- Inference function for external use ---------------------------- #
def inference(model_path, image_path, threshold=0.5):
    """
    Loads a trained model and makes predictions on a single image
    Usage: python train_tumor_model.py --inference --model_path ./outputs/model_inference.pth --image_path ./sample.jpg
    """
    checkpoint = torch.load(model_path)
    class_names = checkpoint['class_names']
    img_size = checkpoint['img_size']
    
    # Create model and load weights
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
        # Image transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    # Load and process image
    results = predict_single_image(model, image_path, transform, class_names, device, threshold)
    
    # Print results
    print(f"\nPrediction results for {Path(image_path).name}:")
    print("-" * 50)
    for pred in results['predictions']:
        result = "POSITIVE" if pred['probability'] > threshold else "NEGATIVE"
        print(f"{pred['class']}: {pred['probability']:.4f} -> {result}")
    
    # Create and save visualization
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(f"Tumor Predictions for {Path(image_path).name}")
    
    # Add prediction text
    pred_text = "\n".join([f"{p['class']}: {p['probability']:.4f} -> {p['prediction']}" 
                           for p in results['predictions']])
    plt.figtext(0.5, 0.01, pred_text, ha="center", fontsize=12, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.axis('off')
    out_dir = Path("./inference_results")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f"pred_{Path(image_path).stem}.png")
    plt.close()
    
    print(f"\nVisualization saved to {out_dir / f'pred_{Path(image_path).stem}.png'}")
    
    return results


# ---------- CLI ------------------------------------------------------------ #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", help="Root OR any of train/valid/test sub‑folders")
    p.add_argument("--out", type=Path, default=Path("./outputs"))
    p.add_argument("--img", type=int, default=384)
    p.add_argument("--bs",  type=int, default=32, help="Batch size (32 fits 8 GB GPU)")
    p.add_argument("--lr",  type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pat", type=int, default=5, help="Early‑stop patience")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed‑precision (recommended for GPU)")
    p.add_argument("--visualize", action="store_true",
                   help="Generate visualizations of test predictions")
    p.add_argument("--num_vis", type=int, default=20,
                   help="Number of test images to visualize")
    
    # Inference mode arguments
    p.add_argument("--inference", action="store_true",
                  help="Run in inference mode on a single image")
    p.add_argument("--model_path", type=str,
                  help="Path to saved model for inference")
    p.add_argument("--image_path", type=str,
                  help="Path to image for inference")
    p.add_argument("--threshold", type=float, default=0.5,
                  help="Threshold for positive prediction")
    
    args = p.parse_args()
    
    if args.inference:
        if not args.model_path or not args.image_path:
            p.error("--inference requires --model_path and --image_path")
        inference(args.model_path, args.image_path, args.threshold)
    else:
        if not args.data_path:
            p.error("--data_path is required for training mode")
        train(args)