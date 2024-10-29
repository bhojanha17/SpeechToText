"""
Train CNN
    Train a convolutional neural network to classify audio samples
    Periodically output training information, and save model checkpoints
    Usage: python train_cnn.py
"""
import torch
from dataset import get_train_val_test_loaders
from model import CNNModel
from common import *
import log
import matplotlib
matplotlib.use('TkAgg') # For showing plots in Ubuntu

# torch.device = torch.device("cpu")


def main():
    """Train CNN and show training plots."""
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(batch_size=32)
    # Model
    model = CNNModel()

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer for learning weights
    optimizer = torch.optim.Adam(params=model.parameters(), lr=(10**-3))

    # Restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(model, "./checkpoints/")

    axes = log.make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
    )

    # initial val loss for early stopping
    global_min_loss = stats[0][1]

    # Patience for early stopping
    patience = 10
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=False,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, "./checkpoints/", stats)

        # update early stopping parameters
        curr_count_to_patience, global_min_loss = early_stopping(
            stats, curr_count_to_patience, global_min_loss
        )

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    log.save_cnn_training_plot()
    log.hold_training_plot()


if __name__ == "__main__":
    main()