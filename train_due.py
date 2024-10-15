import os

import gpytorch
from torch.distributed.elastic.metrics import initialize_metrics

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import torch
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood, GaussianLikelihood
from due import dkl
from due.SpectralNormResNet import SpectralNormResNet
from due.DeepResNet import DeepResNet
from datasets.datasets import pre_dataset, get_spam_or_gamma_dataset
from lib.utils import set_seed, plot_training_history
from lib.uncertainty_metric import UncertaintyMetric
from torch.utils.data import DataLoader
from sngp_wrapper.covert_utils import convert_to_sn_my
import csv
import numpy as np
# nvidia-smi
# GPU_SCORE = torch.cuda.get_device_capability()  -> (8, 9)

NUM_WORKERS = os.cpu_count() #  # <- use all available CPU cores
torch.backends.cuda.matmul.allow_tf32 = True # >= (8, 0)
#
# # Set the device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # Set the device globally
# torch.set_default_device(device)

# total_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
# print(f"Total free GPU memory: {round(total_free_gpu_memory * 1e-9, 3)} GB")
# print(f"Total GPU memory: {round(total_gpu_memory * 1e-9, 3)} GB")

def main(sn_flag = False):

    n_inducing_points = 10
    features = 128
    depth = 6
    coeff = 3. # 0.95 9

    if sn_flag:
        feature_extractor = SpectralNormResNet( input_dim=input_dim, features=features,
                                   depth=depth, spectral_normalization=True, coeff=coeff).cuda()
    else:
        feature_extractor = SpectralNormResNet(input_dim=input_dim, features=features,
                                               depth=depth, spectral_normalization=False, coeff=coeff).cuda()

    # From Tao Wang
    # spec_norm_replace_list = ["Linear", "Conv2D"]
    # spec_norm_bound = 3.  # 9.
    # # # Enforcing Spectral-Normalization on each layer
    # feature_extractor = convert_to_sn_my(feature_extractor, spec_norm_replace_list, spec_norm_bound)

    initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, n_inducing_points
    )

    assert not torch.isnan(initial_inducing_points).any(), "Initial inducing points contain NaN values!"
    assert not torch.isnan(initial_lengthscale).any(), "Initial lengthscale contains NaN values!"

    gp = dkl.GP(
            num_outputs=2,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel="RBF",
        ).cuda()

    model = dkl.DKL(feature_extractor, gp)

    ### For regression tasks
    # likelihood = GaussianLikelihood()
    # https://github.com/cornellius-gp/gpytorch/issues/1001

    # num_features: the number of (independent) features that are output from the GP.
    likelihood = SoftmaxLikelihood( num_features=2, num_classes=num_classes, mixing_weights=False )
    elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    model = model.cuda()
    likelihood = likelihood.cuda()

    lr = 3e-3

    parameters = [
        {"params": model.parameters(), "lr": lr},
    ]
    parameters.append({"params": likelihood.parameters(), "lr": lr})

    optimizer = torch.optim.AdamW(
        parameters
    )

    milestones = [60, 120, 160]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    # For plotting
    plot_train_acc, plot_test_acc = [], []
    plot_train_loss, plot_test_loss = [], []

    def step(engine, batch):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(), y.cuda() #    y ->  torch.Size([64, 2])


        y_pred = model(x)   # MultitaskMultivariateNormal(mean shape: torch.Size([64, 2]))

        loss = loss_fn(y_pred, y)

        with gpytorch.settings.num_likelihood_samples(32):
            y_pred_temp = y_pred.to_data_independent_dist()
            predictive_dist = likelihood(y_pred_temp)
            probs = predictive_dist.probs
            uncertainty = probs.var(0) # (128, 2)

        loss.backward()
        optimizer.step()
        return y, y_pred, loss.item(), uncertainty.detach().cpu().numpy()


    def eval_step(engine, batch):
        model.eval()
        likelihood.eval()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y

    def training_accuracy(output):
        y,y_pred, loss, _ = output
        y_pred = y_pred.to_data_independent_dist()
        y_pred = likelihood(y_pred).probs.mean(0)
        return y_pred, y

    def training_loss(output):
        y_pred, y, loss, _ = output
        return loss

    def output_transform(output):
        y_pred, y = output
        y_pred = y_pred.to_data_independent_dist()
        y_pred = likelihood(y_pred).probs.mean(0)

        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average(output_transform=training_loss)
    metric.attach(trainer, "loss")
    metric = Accuracy(output_transform=training_accuracy)
    metric.attach(trainer, "accuracy")

    # Create an instance of the custom metric
    uncertainty_metric = UncertaintyMetric()
    # Attach it to the trainer engine
    uncertainty_metric.attach(trainer, "uncertainty")

    metric = Accuracy(output_transform=output_transform)
    metric.attach(evaluator, "accuracy")
    metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())
    metric.attach(evaluator, "loss")

    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}

    train_loader = DataLoader( train_dataset, batch_size=128, shuffle=True, drop_last=True, **kwargs )
    print(f"Train loader: {len(train_loader)}")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False, **kwargs )

    @trainer.on(Events.STARTED)
    def initialize_metrics(trainer):
        trainer.state.uncertainty = None


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]
        uncertainty = metrics["uncertainty"]

        print(f"pre_uncertainty: {uncertainty.shape}")

        #global uncertainty

        trainer.state.uncertainty = uncertainty_metric.compute()
        uncertainty_metric.reset()

        print(f"uncertainty: {uncertainty.shape}")


        print(f"Epoch: {trainer.state.epoch} | Train Loss (ELBO): {train_loss:.2f} | Train Acc: {train_acc:.2f} | Uncertainty: {uncertainty}")

        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)


        evaluator.run(test_loader)
        metrics = evaluator.state.metrics

        test_acc = metrics["accuracy"]
        test_loss = metrics["loss"]

        plot_test_acc.append(test_acc)
        plot_test_loss.append(test_loss)

        result = f"Test - Epoch: {trainer.state.epoch} "
        result += f"NLL: {test_loss:.2f} "
        result += f"Acc: {test_acc:.4f} "
        print(result)

        scheduler.step()

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    trainer.run(train_loader, max_epochs= 2)

    final_uncertainty = trainer.state.uncertainty

    # After training, convert accumulated uncertainties to a DataFrame

    print(f"Uncertainty shape: {final_uncertainty.shape}")
    df_uncertainty = pd.DataFrame(final_uncertainty , columns=['uncertainty'])

    # df_all_uncertainties = pd.DataFrame(
    #     np.concatenate(uncertainty, axis=0),  # Concatenate all uncertainties across epochs
    #     columns=['uncertainty']
    # )

    df_uncertainty.to_csv("uncertainties.csv", index=False)

    # df_all_uncertainties["uncertainty"].to_csv("uncertainties.csv", index=False)

    # Done training - time to evaluate
    results = {}
    evaluator.run(test_loader)
    test_acc = evaluator.state.metrics["accuracy"]
    test_loss = evaluator.state.metrics["loss"]

    results["test_accuracy"] = test_acc
    results["test_loss"] = test_loss

    print(f"Final accuracy {results['test_accuracy']:.4f}")

    plot_training_history(plot_train_loss, plot_test_loss, plot_train_acc, plot_test_acc)


    torch.save(model.state_dict(), "model.pt")
    
    if likelihood is not None:
        torch.save(likelihood.state_dict(),  "likelihood.pt")

    test_acc = round(test_acc, 4)


    return test_acc


if __name__ == "__main__":

    seed = 23
    set_seed(seed)
    sn_flag = True

    res = {}
    #data_list = ["Twonorm.arff", "Ring.arff", "Banana.arff", "gamma", "spam"]
    data_list = ["Banana.arff"]

    for data in data_list:
        if data == "gamma" or data == "spam":
            ds = get_spam_or_gamma_dataset(data)
        else:
            ds = pre_dataset(data)

        print(f"Current calculation for {data}")

        input_dim, num_classes, train_dataset, test_dataset = ds
        acc = main(sn_flag)
        res.update( { data:acc } )

    if sn_flag:
        # file_name = "SN_GPIP.json"
        file_name = "SN_GPIP.csv"
    else:
        # file_name = "Final_accuracy_GPIP.json"
        file_name = "GPIP.csv"

    with open(f'{file_name}', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["data", "accuracy"])
        for key, value in res.items():
            writer.writerow([key, value])

