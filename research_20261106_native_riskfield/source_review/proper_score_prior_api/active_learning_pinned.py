import os
from tqdm import tqdm
from typing import Dict
import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn as nn

from source.utils.seeding import fix_seeds
from source.models.utils import variance_link, natural_link, natural_to_gauss
from source.utils.objectives import GaussianNLL
from source.constants import RESULTS_PATH_AL
from source.data.utils import UpsampleDataset, InMemoryDataloader, TensorDataset
from source.trainer import fit

from source.utils.uncertainty_measures import(
    calculate_uncertainties_crps,
    calculate_uncertainties_log,
    calculate_uncertainties_mse,
    calculate_uncertainties_quadratic
)

from source.data.uci import (
    load_ymsd,
    load_sgemm,
    load_ccpp,
    load_casp,
    load_news,
    load_blog,
)

# prediction methods
def predict_ensemble(
        train_loader: InMemoryDataloader,
        val_loader: InMemoryDataloader,
        X_test: torch.Tensor,
        n_samples: int,
        lr: float,
        epochs: int,
        weight_decay: float,
        use_natural: bool,
        device: str
) -> Dict:
    
    means, vars, state_dicts = list(), list(), list()

    # for _ in tqdm(range(n_samples)):
    for _ in range(n_samples):
        model = nn.Sequential(
            nn.Linear(X_test.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        model.to(device)

        model, _, _ = fit(model, train_loader, val_loader, device, epochs, lr, weight_decay, 
                       patience=epochs//5, use_natural=use_natural, verbose=True)

        with torch.no_grad():
            pred = model.forward(X_test).cpu()
            if args.use_natural:
                mean, variance = natural_to_gauss(*natural_link(pred))
            else:
                mean, variance = variance_link(pred)
            means.append(mean)
            vars.append(variance)
        state_dicts.append(model.cpu().state_dict())

    means = torch.stack(means, dim=1)
    vars = torch.stack(vars, dim=1)

    return {
        "means": means,
        "vars": vars,
        "state_dicts": state_dicts,
    }

parser = argparse.ArgumentParser()
# general
parser.add_argument("--dataset", default="ymsd")
parser.add_argument("--method", default="ensemble")
parser.add_argument("--scoring_rule", default="crps")
parser.add_argument("--acquisition_function", default="random")
parser.add_argument("--n_start_samples", default=300, type=int)
parser.add_argument("--n_iterations", default=30, type=int)
parser.add_argument("--n_samples_per_iteration", default=200, type=int)
parser.add_argument("--train_size", default=0, type=int)
    # Note: for now seed only used for everything method specific, dataset uses indep. fixed seed
parser.add_argument("--seed", default=42, type=int) 
parser.add_argument("--device", default="cuda:0")
# method general
parser.add_argument("--use_natural", action="store_true")
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--weight_decay", default=1e-3, type=float)
    # Note: n_samples refers to number of posterior samples (models).
parser.add_argument("--n_samples", default=10, type=int)
# parse
args = parser.parse_args()


# convinience
seed, device = args.seed, args.device
print("Computation executed on >", device)
# if no desired fixed train size is given, use the final train size
final_train_size = args.n_start_samples + (args.n_iterations - 1) * args.n_samples_per_iteration if args.train_size == 0 else args.train_size
print("Train size >", final_train_size)
print("Scoring rule >", args.scoring_rule)
print("Acquisition function >", args.acquisition_function)

# check dataset
assert args.dataset in ["ymsd", "sgemm", "ccpp", "casp", "news", "blog"], f"Dataset {args.dataset} not supported."
# check method
assert args.method in ["ensemble"], f"Method {args.method} not supported."
# check acquisition function
assert args.acquisition_function in ["random", "total_1_1", "total_2_1", "total_3a_1", 
                                     "total_3b_1", "total_3a_2", "total_3b_2",
                                     "bayes_1", "bayes_2", "bayes_3a", "bayes_3b",
                                     "excess_1_1", "excess_2_1", "excess_3a_1", 
                                     "excess_3b_1", "excess_3a_2", "excess_3b_2"], \
                                        f"Acquisition function {args.acquisition_function} not supported."

# define path for results
now = datetime.now().strftime('%Y_%m_%d__%H_%M')
if args.acquisition_function == "random":
    # for random acquisition function, we do not need to save the results
    run_path = os.path.join(RESULTS_PATH_AL, f"{args.dataset}_{args.method}_{args.acquisition_function}_seed{seed}_{now}")
run_path = os.path.join(RESULTS_PATH_AL, f"{args.dataset}_{args.method}_{args.scoring_rule}_{args.acquisition_function}_seed{seed}_{now}")
os.makedirs(run_path, exist_ok=True)

# save command line arguments
formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
with open(os.path.join(run_path, "args.txt"), "w") as file:
    file.write(formatted_args)

# set scoring rule
if args.scoring_rule == "crps":
    calculate_uncertainties = calculate_uncertainties_crps
elif args.scoring_rule == "log":
    calculate_uncertainties = calculate_uncertainties_log
elif args.scoring_rule == "mse":
    calculate_uncertainties = calculate_uncertainties_mse
elif args.scoring_rule == "quadratic":
    calculate_uncertainties = calculate_uncertainties_quadratic
else:
    raise ValueError(f"Scoring rule {args.scoring_rule} not supported.")

#################
### LOAD DATA ###
#################

if args.dataset == "ymsd":
    load_dataset = load_ymsd
elif args.dataset == "sgemm":
    load_dataset = load_sgemm
elif args.dataset == "ccpp":
    load_dataset = load_ccpp
elif args.dataset == "casp":
    load_dataset = load_casp
elif args.dataset == "news":
    load_dataset = load_news 
elif args.dataset == "blog":
    load_dataset = load_blog
else:
    raise ValueError(f"Dataset {args.dataset} not supported.")

# load dataset
X_train, y_train, X_test, y_test = load_dataset()

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32)

# save test targets for evaluation
torch.save(y_test, os.path.join(run_path, "y_test.pt"))

rng = np.random.default_rng(seed=42) # dataset split constant over runs
val_inds = rng.choice(np.arange(len(X_train)), size=1_000, replace=False)
train_inds = torch.as_tensor(np.delete(np.arange(len(X_train)), val_inds))

full_train_ds = TensorDataset(X_train[train_inds], y_train[train_inds])
val_ds = TensorDataset(X_train[val_inds], y_train[val_inds])

#################
### MAIN LOOP ###
#################

test_perfs = list()
# seeding
fix_seeds(seed)

for i in range(args.n_iterations):

    print(f"Iteration {i + 1} / {args.n_iterations}")

    if i == 0:
        indices = np.arange(len(full_train_ds))
        rng.shuffle(indices)
        indices = torch.as_tensor(indices[:args.n_start_samples])
        remaining_indices = torch.as_tensor(np.delete(np.arange(len(full_train_ds)), indices.numpy()))
        train_ds = TensorDataset(full_train_ds.inputs[indices], full_train_ds.targets[indices])
        pool_ds = TensorDataset(full_train_ds.inputs[remaining_indices], full_train_ds.targets[remaining_indices])
    else:
        train_ds = TensorDataset(torch.cat((train_ds.inputs, pool_ds.inputs[acquired_samples_indices]), dim=0),
                                 torch.cat((train_ds.targets, pool_ds.targets[acquired_samples_indices]), dim=0))
        remaining_indices = torch.as_tensor(np.delete(np.arange(len(pool_ds)), acquired_samples_indices.cpu().numpy()))
        pool_ds = TensorDataset(pool_ds.inputs[remaining_indices], pool_ds.targets[remaining_indices])

    print(len(train_ds), len(pool_ds))

    # create dataloader for training
    train_loader = InMemoryDataloader(UpsampleDataset(train_ds, upsample=final_train_size),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = InMemoryDataloader(val_ds, batch_size=len(val_ds), shuffle=False)
    # create tensor for pool
    X_pool = pool_ds.inputs


    # predict ensemble
    results = predict_ensemble(train_loader, 
                                val_loader, 
                                torch.cat((X_pool, X_test), dim=0),
                                args.n_samples, 
                                args.lr, 
                                args.epochs, 
                                args.weight_decay,
                                args.use_natural,
                                device)
    
    pool_means = results["means"][:len(pool_ds)]
    pool_vars = results["vars"][:len(pool_ds)]

    # calculate uncertainties
    if args.acquisition_function != "random":
        uncertainties = calculate_uncertainties(pool_means, pool_vars)
        uncertainties = uncertainties[f"{args.acquisition_function}"]
    else:
        uncertainties = torch.rand(size=(len(pool_means), ))
    
    # _ , acquired_samples_indices = torch.topk(uncertainties, k=args.n_samples_per_iteration, dim=0, largest=True, sorted=False)
    acquired_samples_indices = torch.multinomial(torch.softmax(uncertainties, dim=0), 
                                                 num_samples=args.n_samples_per_iteration, 
                                                 replacement=False)

    test_means = results["means"][-len(X_test):].cpu()
    test_vars = results["vars"][-len(X_test):].cpu()
    # calculate mse for checking
    # test_perf = torch.mean((test_means.mean(dim=1) - y_test) ** 2).item()
    # calculate gaussian log likelihood for checking
    metric = GaussianNLL()
    test_perf = metric(test_means, test_vars, y_test.unsqueeze(1)).item()

    print(f"test performance: {test_perf:.03f}")
    test_perfs.append(test_perf)
    
    # save test preds and state dicts
    torch.save(test_means.to(dtype=torch.float16), os.path.join(run_path, f"means_{i}.pt"))
    torch.save(test_vars.to(dtype=torch.float16), os.path.join(run_path, f"vars_{i}.pt"))
    torch.save(results["state_dicts"], os.path.join(run_path, f"state_dicts_{i}.pt"))

    # save test performances as text file
    with open(os.path.join(run_path, "test_perfs.txt"), "w") as file:
        file.write("\n".join(str(perf) for perf in test_perfs))

