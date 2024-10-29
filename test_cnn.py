"""
Test CNN
    Test our trained CNN from train_cnn.py on the test data.
    Load the trained CNN model from a saved checkpoint and evaulate using
    accuracy and AUROC metrics.
    Usage: python test_cnn.py
"""

import torch
from dataset import get_train_val_test_loaders
from model import CNNModel
from common import *
import log
import librosa
from common import predictions
import sounddevice as sd
import soundfile as sf
import matplotlib
matplotlib.use('TkAgg') # For showing plots in Ubuntu


def main():
    """Print performance metrics for model at specified epoch."""
    # Data loaders
    tr_loader, va_loader, te_loader, label_dict = get_train_val_test_loaders(batch_size=32)

    # Model
    model = CNNModel()

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(model, "./checkpoints/")

    # Let user select mode
    inp = ""
    while inp != "data" and inp != "audio" and inp != "exit":
        print("Use test data or audio input? [data, audio, exit]")
        inp = str(input())
    if inp == "exit":
        return
    if inp == "audio":
        filename = "test_audio.wav"
        print("start")
        mydata = sd.rec(16000, samplerate=16000, channels=1, blocking=True)
        print("end")
        sd.wait()
        sf.write("./input/" + filename, mydata, 16000)
        yes_samples = librosa.load("./input/" + filename, sr = 8000)[0].reshape(1, 1, 8000)
        output = model(torch.from_numpy(yes_samples))
        pred = predictions(output.data)
        print("Le prediction:", label_dict(pred[0]))
        return

    axes = log.make_training_plot()

    # Evaluate the model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()