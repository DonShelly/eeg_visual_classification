import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from models.lstm import Model


torch.utils.backcompat.broadcast_warning.enabled = True
cudnn.benchmark = True

splitsPath = os.path.abspath("../../datasets/original_data/block_splits_by_image.pth")
eegDataset = os.path.abspath("../../datasets/original_data/eeg_signals_128_sequential_band_all_with_mean_std.pth")

time_low = 320
time_high = 480


# Dataset loading class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)

        self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[time_low:time_high, :]

        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label


def load_data(data_dir="../../datasets/original_data/eeg_signals_128_sequential_band_all_with_mean_std.pth"):
    # Load dataset
    dataset = EEGDataset(data_dir)
    # Create loaders
    loaders = {split: DataLoader(Splitter(dataset, split_path=splitsPath, split_num=0, split_name=split), batch_size=16, drop_last=True, shuffle=True) for split in ["train", "val", "test"]}

    return loaders


def train_classifier(config, checkpoint_dir=None, data_dir=None):
    # print("model options: " + model_options)

    # Load model
    model = Model(config["input_size"], config["lstm_size"], config["lstm_layers"], config["output_size"])

    # Setup CUDA
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # load data
    loaders = load_data(data_dir=data_dir)

# ####KKKEEEEEEEEEEEPPPPP~~~~~'########'########
    # if opt.pretrained_net != '':
    #     model = torch.load(opt.pretrained_net)
    #     print(model)
    ###############################################

    # initialize training,validation, test losses and accuracy list
    # losses_per_epoch = {"train": [], "val": [], "test": []}
    # accuracies_per_epoch = {"train": [], "val": [], "test": []}

    # best_accuracy = 0
    # best_accuracy_val = 0
    # best_epoch = 0

    # Start training
    for epoch in range(50):
        # Initialize loss/accuracy variables
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}

        # Process each split
        # for split in ("train", "val", "test"):
        for split in ("train", "val"):
            # Set network mode
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)
            # Process all split batches
            for i, (input, target) in enumerate(loaders[split]):
                # Check CUDA
                input = input.to(device)
                target = target.to(device)
                # Forward
                output = model(input)

                # Compute loss
                loss = F.cross_entropy(output, target)
                losses[split] += loss.item()
                # Compute accuracy
                _, pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracy = correct / input.data.size(0)
                accuracies[split] += accuracy
                counts[split] += 1
                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        #Print info at the end of the epoch
        # if accuracies["val"] / counts["val"] >= best_accuracy_val:
        #     best_accuracy_val = accuracies["val"] / counts["val"]
        #     best_accuracy = accuracies["test"] / counts["test"]
        #     best_epoch = epoch

        # TrL, TrA, VL, VA, TeL, TeA = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses["val"] / counts["val"], accuracies["val"] / counts["val"], losses["test"] / counts["test"], accuracies["test"] / counts["test"]
        # print("Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [?-? Hz] - Epoch {0}: TrL={1:.4f}, "
        #       "TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {"
        #       "8:d}".format(epoch,
        #                     losses["train"] / counts["train"],
        #                     accuracies["train"] / counts["train"],
        #                     losses["val"] / counts["val"],
        #                     accuracies["val"] / counts["val"],
        #                     losses["test"] / counts["test"],
        #                     accuracies["test"] / counts["test"],
        #                     best_accuracy, best_epoch, time_low, time_high, "lstm", 0))

        # losses_per_epoch['train'].append(TrL)
        # losses_per_epoch['val'].append(VL)
        # losses_per_epoch['test'].append(TeL)
        # accuracies_per_epoch['train'].append(TrA)
        # accuracies_per_epoch['val'].append(VA)
        # accuracies_per_epoch['test'].append(TeA)

        # if epoch % opt.saveCheck == 0:
        #     torch.save(model, '%s__subject%d_epoch_%d.pth' % (opt.model_type, opt.subject, epoch))

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=losses["val"] / counts["val"], accuracy=accuracies["val"] / counts["val"])
    print("Finished Training")


# default device is always CPU for compatability
def test_accuracy(model, device="cpu"):

    loaders = load_data()
    testloader = loaders["test"]

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            eeg, labels = data
            eeg, labels = eeg.to(device), labels.to(device)
            outputs = model(eeg)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



# print("config: " + str(config))


def main(num_samples=10, max_time=10, gpus_per_trial=1, cpus_per_trial=2, num_of_epochs=10):

    data_dir = os.path.abspath("../../datasets/original_data/eeg_signals_128_sequential_band_all_with_mean_std.pth")

    config = {
        "input_size": 128,
        "lstm_size": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "lstm_layers": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "output_size": 128,
        "lr": tune.loguniform(1e-4, 1e-1)
        # "batch_size": tune.choice([2, 4, 8, 16])
    }

    scheduler = ASHAScheduler(
        max_t=max_time,
        grace_period=1,
        reduction_factor=2)

    result = tune.run(
        tune.with_parameters(train_classifier, data_dir=data_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = Model(layers1=best_trial.config["layers1"], best_trial.config["l2"])
    best_trained_model = Model(lstm_layers=best_trial.config["lstm_layers"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_time=200, gpus_per_trial=1, cpus_per_trial=20, num_of_epochs=50)
