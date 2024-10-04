import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
from torch.utils.data import DataLoader

from sngp_wrapper.covert_utils import convert_to_sn_my
import json

# nvidia-smi
NUM_WORKERS = os.cpu_count()

def main(sn_flag = False):

    #### spam or gamma
    # ds = get_spam_or_gamma_dataset("gamma")

    #### input: "Twonorm.arff",X.shape: (7400, 20) "Ring.arff", (7400, 20) "Banana.arff" (5300,2)
    # ds =  pre_dataset("Banana.arff")
    # input_dim, num_classes, train_dataset, test_dataset = ds

    n_inducing_points = 10

    # feature_extractor = DeepResNet(input_dim = 2,  num_layers = 3, num_hidden = 128,
    # activation = "relu", num_outputs = None, dropout_rate = 0.1) # (128 , 128)

    features = 128
    depth = 3
    coeff = 3. # 0.95

    if sn_flag:
        feature_extractor = SpectralNormResNet( input_dim = input_dim, features = features,
                                   depth = depth, spectral_normalization = True, coeff = coeff).cuda()
    else:
        feature_extractor = SpectralNormResNet(input_dim = input_dim, features = features,
                                               depth = depth, spectral_normalization = False, coeff = coeff ).cuda()

    # spec_norm_replace_list = ["Linear", "Conv2D"]
    # spec_norm_bound = 3.  # 9.
    # # # Enforcing Spectral-Normalization on each layer
    # feature_extractor = convert_to_sn_my(feature_extractor, spec_norm_replace_list, spec_norm_bound)

    initial_inducing_points, initial_lengthscale = dkl.initial_values(
            train_dataset, feature_extractor, n_inducing_points
    )   # (10, 128)

    assert not torch.isnan(initial_inducing_points).any(), "Initial inducing points contain NaN values!"
    assert not torch.isnan(initial_lengthscale).any(), "Initial lengthscale contains NaN values!"

    gp = dkl.GP(
            num_outputs = 2,
            initial_lengthscale = initial_lengthscale,
            initial_inducing_points = initial_inducing_points,
            kernel = "RBF",
        ).cuda()

    model = dkl.DKL(feature_extractor, gp)

    ### For regression tasks
    # likelihood = GaussianLikelihood()

    # https://github.com/cornellius-gp/gpytorch/issues/1001

    # num_features: the number of (independent) features that are output from the GP.
    likelihood = SoftmaxLikelihood( num_features = 2, num_classes = num_classes, mixing_weights = False )

    elbo_fn = VariationalELBO(likelihood, gp, num_data = len(train_dataset))

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
        optimizer, milestones = milestones, gamma=0.2
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
        loss.backward()
        optimizer.step()
        return y, y_pred, loss.item()


    def eval_step(engine, batch):
        model.eval()
        likelihood.eval()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            y_pred = model(x)
        return y_pred, y

    def training_accuracy(output):
        y,y_pred, loss = output
        y_pred = y_pred.to_data_independent_dist()
        y_pred = likelihood(y_pred).probs.mean(0)
        return y_pred, y

    def training_loss(output):
        y_pred, y, loss = output
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

    metric = Accuracy(output_transform = output_transform)
    metric.attach(evaluator, "accuracy")

    metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())
    metric.attach(evaluator, "loss")

    kwargs = {"num_workers": NUM_WORKERS, "pin_memory": True}

    train_loader = DataLoader( train_dataset, batch_size = 128, shuffle=True, drop_last = True, **kwargs )
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle=True, drop_last = True, **kwargs )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]

        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)

        result = f"Train - Epoch: {trainer.state.epoch} "
        result += f"ELBO: {train_loss:.2f} "
        result += f"Accuracy: {train_acc :.2f} "
        print(result)

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

    pbar = ProgressBar(dynamic_ncols = True)
    pbar.attach(trainer)

    trainer.run(train_loader, max_epochs = 50)
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

    res = {}
    data_list = ["Twonorm.arff", "Ring.arff", "Banana.arff", "gamma", "spam"]
    for data in data_list:
        if data == "gamma" or data == "spam":
            ds = get_spam_or_gamma_dataset(data)
        else:
            ds = pre_dataset(data)
        input_dim, num_classes, train_dataset, test_dataset = ds
        acc = main()

        res.update( { data:acc } )

    file_name = "Final_accuracy.json"
    # Open the file in write mode and use json.dump() to write the JSON data
    with open(file_name, 'w') as file:
        json.dump(res, file, indent = 4)

    print( json.dumps(res) )