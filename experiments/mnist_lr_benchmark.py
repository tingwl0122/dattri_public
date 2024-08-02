"""This experiment benchmarks TDA methods on the MNIST-10 dataset."""

# ruff: noqa
import argparse
from functools import partial

import numpy as np
import torch
from torch import nn

from dattri.algorithm.influence_function import (
    IFAttributorArnoldi,
    IFAttributorCG,
    IFAttributorDataInf,
    IFAttributorExplicit,
    IFAttributorLiSSA,
)
from dattri.algorithm.rps import RPSAttributor
from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import (
    create_mnist_dataset,
)
from dattri.benchmark.datasets.mnist.mnist_lr import (
    create_lr_model,
    loss_mnist_lr,
    train_mnist_lr,
)
from dattri.benchmark.utils import SubsetSampler
from dattri.metrics.ground_truth import calculate_lds_ground_truth
from dattri.metrics.metrics import lds
from dattri.task import AttributionTask

IHVP_CONFIG = {
    "explicit": [
        {"regularization": r} for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "cg": [
        {"regularization": r, "max_iter": 10}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "lissa": [
        {"recursion_depth": 100, "batch_size": 100},
        {"recursion_depth": 100, "batch_size": 50},
        {"recursion_depth": 100, "batch_size": 10},
    ],
    "arnoldi": [
        {"regularization": r, "max_iter": 100, "proj_dim": 1000}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="mnist")
    argparser.add_argument("--model", type=str, default="lr")
    argparser.add_argument("--method", type=str, default="cg")
    args = argparser.parse_args()

    print(args)
    # create dataset
    dataset_train, dataset_test = create_mnist_dataset(
        "./data",
    )

    # the exp size
    train_size = 5000
    test_size = 500
    # train/test dataloader
    train_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        sampler=SubsetSampler(range(train_size)),
    )
    train_loader_full = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_size,
        sampler=SubsetSampler(range(train_size)),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_size,
        sampler=SubsetSampler(range(train_size)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_size,
        sampler=SubsetSampler(range(test_size)),
    )

    # the eval size
    lds_eval_size = 50
    # record best configs and results
    best_result = 0
    best_config = None
    IF_MAP = {
        "explicit": IFAttributorExplicit,
        "cg": IFAttributorCG,
        "lissa": IFAttributorLiSSA,
        "arnoldi": IFAttributorArnoldi,
        "datainf": IFAttributorDataInf,
    }
    if args.method in ["explicit", "cg", "lissa", "arnoldi", "datainf"]:
        for ihvp_config in IHVP_CONFIG[args.method]:
            # define model
            model = train_mnist_lr(train_loader_train, device="cuda")

            # define target function
            def f(params, data_target_pair):
                image, label = data_target_pair
                loss = nn.CrossEntropyLoss()
                yhat = torch.func.functional_call(model, params, image)
                return loss(yhat, label.long())

            # define task
            task = AttributionTask(
                model=model,
                loss_func=f,
                checkpoints=model.state_dict(),
            )
            # define attributor
            attributor = IF_MAP[args.method](
                task=task,
                device="cuda",
                **ihvp_config,
            )
            attributor.cache(train_loader_full)
            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(train_loader, test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            # get retrained models' location
            retrain_dir = "/home/shared/dattri-dataset/mnist_lds_test/models"
            # get model output and indices
            target_values, indices = calculate_lds_ground_truth(
                partial(loss_mnist_lr, device="cuda"),
                retrain_dir,
                test_loader,
            )

            # compute LDS value
            lds_score_full = lds(score.T.cpu(), (-target_values, indices))[0]

            sum_val = 0
            counter = 0
            for i in range(lds_eval_size):
                if np.isnan(lds_score_full[i]):
                    continue
                sum_val += lds_score_full[i]
                counter += 1
            print("config: ", ihvp_config)
            print("LDS value: ", sum_val / counter)

            if sum_val / counter > best_result:
                best_result = sum_val / counter
                best_config = ihvp_config

        print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "TRAK":
        for proj_dim, ensemble in [(512, 1), (512, 10), (512, 50)]:
            # define model
            model = create_lr_model("mnist").cuda()
            # define checkpoint locations
            ckpts = []
            for i in range(ensemble):
                ckpts.append(
                    f"/home/shared/dattri-dataset/mnist_lds_test/models/{i}/model_weights_0.pt",
                )

            # define target functions
            def f_0(params, data_target_pair):
                image, label = data_target_pair
                image_t = image.unsqueeze(0)
                label_t = label.unsqueeze(0)
                loss = nn.CrossEntropyLoss()
                yhat = torch.func.functional_call(model, params, image_t)
                logp = -loss(yhat, label_t)
                return logp - torch.log(1 - torch.exp(logp))

            def m(params, image_label_pair):
                image, label = image_label_pair
                image_t = image.unsqueeze(0)
                label_t = label.unsqueeze(0)
                loss = nn.CrossEntropyLoss()
                yhat = torch.func.functional_call(model, params, image_t)
                p = torch.exp(-loss(yhat, label_t.long()))
                return p

            # define task
            projector_kwargs = {
                "proj_dim": proj_dim,
                "device": "cuda",
            }
            task = AttributionTask(
                model=create_lr_model("mnist").cuda(),
                loss_func=f_0,
                checkpoints=ckpts,
            )

            # define attributor
            attributor = TRAKAttributor(
                task=task,
                correct_probability_func=m,
                device="cuda",
                projector_kwargs=projector_kwargs,
            )

            attributor.cache(train_loader)
            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            # get retrained models' location
            retrain_dir = "/home/shared/dattri-dataset/mnist_lds_test/models"
            # get model output and indices
            target_values, indices = calculate_lds_ground_truth(
                partial(loss_mnist_lr, device="cuda"),
                retrain_dir,
                test_loader,
            )
            # compute LDS value
            lds_score_full = lds(score.T.cpu(), (-target_values, indices))[0]

            lds_value = torch.mean(lds_score_full[~torch.isnan(lds_score_full)])
            print(proj_dim, ensemble)
            print("lds:", lds_value)
            if lds_value > best_result:
                best_result = lds_value
                best_config = (proj_dim, ensemble)

        print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "TracIn":
        for ensemble, normalized_grad in [(1, True), (1, False), (10, False)]:
            # define model
            model = create_lr_model("mnist").cuda()
            ckpts = []
            for i in range(ensemble):
                ckpts.append(
                    f"/home/shared/dattri-dataset/mnist_lds_test/models/{i}/model_weights_0.pt",
                )

            # define target function
            def f_tracin(params, data_target_pair):
                image, label = data_target_pair
                image_t = image.unsqueeze(0)
                label_t = label.unsqueeze(0)
                loss = nn.CrossEntropyLoss()
                yhat = torch.func.functional_call(model, params, image_t)
                return loss(yhat, label_t.long())

            # define task
            proj_kwargs = {
                "proj_dim": 512,
                "proj_max_batch_size": 32,
                "proj_seed": 0,
                "device": "cuda",
                "use_half_precision": False,
            }
            task = AttributionTask(
                model=model,
                loss_func=f_tracin,
                checkpoints=ckpts,
            )

            # define attributor
            attributor = TracInAttributor(
                task=task,
                weight_list=torch.ones(ensemble),
                normalized_grad=normalized_grad,
                device="cuda",
            )

            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(train_loader, test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            # get retrained models' location
            retrain_dir = "/home/shared/dattri-dataset/mnist_lds_test/models"
            # get model output and indices
            target_values, indices = calculate_lds_ground_truth(
                partial(loss_mnist_lr, device="cuda"),
                retrain_dir,
                test_loader,
            )
            # compute LDS value
            lds_score_full = lds(score.T.cpu(), (-target_values, indices))[0]

            sum_val = 0
            counter = 0
            for i in range(lds_eval_size):
                if np.isnan(lds_score_full[i]):
                    continue
                sum_val += lds_score_full[i]
                counter += 1
            print("config: ", (ensemble, normalized_grad))
            print("LDS value: ", sum_val / counter)
            if sum_val / counter > best_result:
                best_result = sum_val / counter
                best_config = (ensemble, normalized_grad)

        print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "RPS":
        for l2 in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
            for normalize_pre in [True, False]:
                model = create_lr_model("mnist").cuda()
                model.load_state_dict(
                    torch.load(
                        "/home/shared/dattri-dataset/mnist_lds_test/models/0/model_weights_0.pt"
                    )
                )

                def f_rps(pre_activation_list, label_list):
                    loss_fn = nn.CrossEntropyLoss()
                    return loss_fn(pre_activation_list, label_list)

                # define task
                task = AttributionTask(
                    model=model,
                    loss_func=f_rps,
                    checkpoints=model.state_dict(),
                )

                # define attributor
                attributor = RPSAttributor(
                    task=task,
                    final_linear_layer_name="linear",
                    nomralize_preactivate=normalize_pre,
                    l2_strength=l2,
                    device="cuda",
                )

                torch.cuda.reset_peak_memory_stats("cuda")
                score = attributor.attribute(train_loader, test_loader)
                peak_memory = (
                    torch.cuda.max_memory_allocated("cuda") / 1e6
                )  # Convert to MB
                print(f"Peak memory usage: {peak_memory} MB")

                # get retrained models' location
                retrain_dir = "/home/shared/dattri-dataset/mnist_lds_test/models"
                # get model output and indices
                target_values, indices = calculate_lds_ground_truth(
                    partial(loss_mnist_lr, device="cuda"),
                    retrain_dir,
                    test_loader,
                )
                # compute LDS value
                lds_score_full = lds(score.T.cpu(), (-target_values, indices))[0]

                sum_val = 0
                counter = 0
                for i in range(lds_eval_size):
                    if np.isnan(lds_score_full[i]):
                        continue
                    sum_val += lds_score_full[i]
                    counter += 1
                print("config: ", (l2, normalize_pre))
                print("LDS value: ", sum_val / counter)
                if sum_val / counter > best_result:
                    best_result = sum_val / counter
                    best_config = (l2, normalize_pre)

        print(args.method, "RESULT:", best_config, "lds:", best_result)
