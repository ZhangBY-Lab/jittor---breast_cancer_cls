import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from _cfg import cfg
from _loss import CombinedBCEFocalLoss
from _model import CancerModel
from _datasets import ImageFolder, build_transform
from _utils import device_usr, create_dir, initialize_random_states


# ============== Helper function to print logs ==============
def log_progress(epoch, num_epochs, losses, optimizer_lr):
    """Print the training progress for each epoch."""
    avg_loss = np.mean(losses)
    print(f'Epoch {epoch} / {num_epochs} [TRAIN] mean loss = {avg_loss:.4f}')
    print(f"Current Learning Rate: {optimizer_lr:.6f}")


# ============== Training loop with progress bar ==============
def train_one_epoch(model, optimizer, train_loader, loss_fn, epoch, num_epochs):
    model.train()
    train_loader.set_epoch(epoch)
    losses = []

    pbar = tqdm(train_loader, total=len(train_loader),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for data in pbar:
        image, label = data
        # print(image.shape)
        pred = model(image)
        loss = loss_fn(pred, label)
        loss.sync()
        optimizer.step(loss)

        # Store loss and update progress bar description
        losses.append(loss.item())
        pbar.set_description(f'Epoch {epoch} [TRAIN] loss = {losses[-1]:.4f}')

    # Log the progress at the end of the epoch
    log_progress(epoch, num_epochs, losses, optimizer.state_dict()['defaults']['lr'])


# ============== Evaluation Function ==============
def evaluate(model: nn.Module, val_loader: Dataset):
    model.eval()
    sigmoid = nn.Sigmoid()
    preds, targets = [], []
    print("Evaluating...")

    for data in val_loader:
        image, label = data
        pred = model(image)
        pred.sync()

        prob = sigmoid(pred).numpy()
        true_labels = np.argmax(label.numpy(), axis=1)

        preds.append(prob.argmax(axis=1))
        targets.append(true_labels)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    acc = accuracy_score(targets, preds)
    return acc


# ============== Training Loop and Saving Model ==============
def run_training_loop(model, optimizer, scheduler, train_loader, val_loader, loss_fn, num_epochs, model_dir, fold):
    best_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, loss_fn, epoch, num_epochs)

        acc = evaluate(model, val_loader)
        scheduler.step(acc)

        # Save the model with the best accuracy
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            model_path = os.path.join(model_dir, f'fold{fold}_best_acc:{best_acc}_epoch{epoch}.pkl')
            model.save(model_path)
        
        model.save(os.path.join(model_dir, f'fold{fold}_epoch{epoch}.pkl'))

        print(f'Epoch {epoch} - acc: {acc:.4f} | Best acc: {best_acc:.4f} | Best epoch: {best_epoch}')


# ============== Main Function ==============
def main():
    initialize_random_states(cfg.seed)
    create_dir(cfg.model_dir)

    transform = build_transform()

    for fold in cfg.folds:
        print(f"====================== Fold {fold} ======================")

        # Initialize Model
        model = CancerModel(
            model_name=cfg.backbone,
            model_name2=cfg.backbone2,
            num_classes=6,
            pretrain=True,
            pooling_type='gem',
            dropout_rate=0.4
        )

        # Load Dataset
        df = pd.read_csv(cfg.csv_dir)
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]

        # Initialize Optimizer, Scheduler, and Loss Function
        optimizer = jt.optim.Adan(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # optimizer = jt.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        scheduler = jt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=6, verbose=True)

        loss_fn = CombinedBCEFocalLoss(
            focal_kwargs={
                'alpha': 0.25,
                'gamma': 2.0,
                'adaptive_gamma': False,
                'smooth_focus': False
            },
            adaptive_weight=False,
            size_average=True
        )

        # Initialize DataLoaders
        train_loader = ImageFolder(
            root=cfg.dataroot,
            df=train_df,
            transform=transform,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            num_classes=cfg.num_classes,
            shuffle=True,
            mode="train"
        )

        val_loader = ImageFolder(
            root=cfg.dataroot,
            df=val_df,
            transform=transform,
            batch_size=cfg.batch_size_val,
            num_workers=cfg.num_workers,
            num_classes=cfg.num_classes,
            shuffle=False,
            mode="val"
        )

        run_training_loop(model, optimizer, scheduler, train_loader, val_loader, loss_fn, cfg.epochs, cfg.model_dir,
                          fold)


if __name__ == '__main__':
    main()
