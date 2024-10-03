import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import h5py
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import argparse
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian
# warnings.filterwarnings('error')
import operator
import wandb
from functools import partial
import copy



def none_or_float(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float or 'None'")


def convert_threshold(value):
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold value: {value}")


def get_abmil_params():
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--trial', type=int, default=4, help='Set trial')
    parser.add_argument('--fold', type=int, default=5, help='Set fold')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--model_name', type=str, default="uni", help='Model name')
    parser.add_argument('--spec_norm_bound', type=none_or_float, default=None,
                        help='Spectral norm bound, if set not None, will use spectral normalization')
    parser.add_argument('--gaussian_process', action='store_true',
                        help='If set True, will use Laplace Approximization of Gaussian process to estimate the uncertainty')
    parser.add_argument('--gp_num_inducing', type=int, default=None,
                        help='Number of inducing points for Gaussian process')

    parser.add_argument('--spec_norm_replace_list', type=str, default='Linear,Conv2D',
                        help='List of layers to replace with spectral normalization')
    parser.add_argument('--save_to_parquet', action='store_true',
                        help='If set True, will save the results to parquet file')
    parser.add_argument('--save_destination', type=str, default="/home/user/sngp/UniConch/models/",
                        help='Model and parquet save path')
    parser.add_argument('--mask_tile', action='store_true', help='whether to mask the tiles')
    parser.add_argument('--mask_tile_category', type=str, default="rand",
                        choices=["rand", "in_slide", "in_slide_weight", "all_slide"], help='whether to mask the tiles')
    parser.add_argument('--mask_tile_threshold', type=convert_threshold, default=4, help='mask tile threshold')
    parser.add_argument('--invert_threshold', action='store_true', help='whether to invert the threshold')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate the model')
    parser.add_argument('--shrink_etest', action='store_true', help='shrink the etest dataset only for tunin_method=2')
    parser.add_argument('--data_specific_evaluation', action='store_true', help='itest / etest from different tuning')

    parser.add_argument('--hyperopt', action='store_true', help='whether to search the hyperparameters')

    parser.add_argument('--results_file_path', type=str, default="final_results.csv", help='final_results.csv')
    parser.add_argument('--tuning_method', type=int, default=0, help='tuning method')
    parser.add_argument('--force_middle_prefix', type=str, default=None, help='force middle prefix')
    parser.add_argument('--set', dest='set_gp', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()
    args.spec_norm_replace_list = args.spec_norm_replace_list.split(',')
    args.save_destination = Path(args.save_destination)
    if args.shrink_etest:
        assert args.tuning_method == 2, "shrink_etest should only be used for tuning_method=3"
    if args.data_specific_evaluation:
        assert args.shrink_etest, "data_specific_evaluation should only be used with shrink_etest"
    assert args.results_file_path.endswith(
        ".csv"), f"results_file_path should be a csv file, get {args.results_file_path}"
    return parser, args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle_model(path: str = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def save_pickle_data(data, path: str = "kmeans_100w.pkl"):
    with open(path, "wb") as file:
        pickle.dump(data, file)


class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMIL, self).__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 2)

    def forward(self, H):
        tanh_VH = torch.tanh(self.V(H))
        attention_scores = self.w(tanh_VH)
        attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights


class GatedAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_input, dropout_hidden):
        super(GatedAttention, self).__init__()
        assert 0 <= dropout_input <= 1 and 0 <= dropout_hidden <= 1
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_input))
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_hidden))
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        c = a.mul(b)
        c = self.w(c)
        prob = F.softmax(c, dim=1)  # abmil likes to use batch size 1
        return (prob * x).sum(dim=1)


class GatedABMIL(nn.Module):
    """https://github.com/mahmoodlab/CLAM/models/model_clam.py
       The only differences is that we use single mapping to enable uni and conch with the same hidden state for attention
    """

    def __init__(self, embed_dim: int = 1024, hdim1: int = 512, hdim2: int = 384, n_classes: int = 2):
        super(GatedABMIL, self).__init__()
        if embed_dim == 512:
            self.fair_proj = nn.Linear(embed_dim, 1024)
            print("use fair projection")
            embed_dim = 1024
        else:
            self.fair_proj = nn.Identity()
        self.feature_extractor = nn.Sequential(nn.Linear(embed_dim, hdim1), nn.ReLU(), nn.Dropout(0.1))
        self.attention_layer = GatedAttention(hdim1, hdim2, dropout_input=0.25, dropout_hidden=0.25)
        self.classifier = nn.Linear(hdim1, n_classes)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, **kwargs):
        x = self.fair_proj(x)
        x = self.feature_extractor(x)
        x = self.attention_layer(x)
        return self.classifier(x, **kwargs)


def amb_function(model_name, dataset_name, tuning_method):
    "conch_etest_ambiguity_dict_autogluon_0.2_tuning0.pkl"
    return f"/home/user/sngp/UniConch/models/ambpkl/newambk/{model_name}_{dataset_name}_ambiguity_dict_autogluon_0.2_tuning{tuning_method}.pkl"
    # return f"/home/user/sngp/UniConch/models/ambpkl/{model_name}_{dataset_name}_ambiguity_dict_autogluon_0.1.pkl"
    # return f"/home/user/sngp/UniConch/models/ambpkl/{model_name}_{dataset_name}_ambiguity_dict_lightgbm_0.2.pkl"
    # return f"/home/user/sngp/UniConch/models/ambpkl/{model_name}_{dataset_name}_ambiguity_dict_lightgbm_0.5.pkl"


class UniTrainingConfig:
    model_name = "uni"
    dataset_h5 = Path("/home/user/sngp/UniConch/uni_tcga_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = Path("/home/user/sngp/UniConch/uni_cptac_h5file")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")
    mask_func = amb_function

    embed_dim = 1024
    batch_size = 1
    num_workers = 8
    epochs = 20
    label_dict = {"LUAD": 0, "LUSC": 1}


class ConchTrainingConfig:
    model_name = "conch"
    dataset_h5 = Path("/home/user/sngp/UniConch/conch_tcga_h5file")
    etest_h5 = Path("/home/user/sngp/UniConch/conch_cptac_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")
    mask_func = amb_function

    embed_dim = 512
    batch_size = 1
    num_workers = 8
    epochs = 20
    label_dict = {"LUAD": 0, "LUSC": 1}


# GP_KWARGS_CONCH = {
#     'num_inducing': 512,
#     'gp_scale': 1.0,
#     'gp_bias': 0.,
#     'gp_kernel_type': 'linear',
#     'gp_input_normalization': False,
#     'gp_cov_discount_factor': -1,
#     'gp_cov_ridge_penalty': 1.,
#     'gp_scale_random_features': False,
#     'gp_use_custom_random_features': True,
#     'gp_random_feature_type': 'orf',
#     'gp_output_imagenet_initializer': True,
# }
GP_KWARGS_CONCH = {
    'num_inducing': 512,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'linear',
    'gp_input_normalization': False,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': True,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': False,
}

GP_KWARGS_UNI = {
    'num_inducing': 2048,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'gaussian',
    'gp_input_normalization': True,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': False,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': True,
}




class InceptionTrainingConfig:
    model_name = "inception-sngp"
    persistant_h5 = Path("/home/user/sngp/images/h5_sngp_feature")
    dataset_h5 = None
    etest_h5 = None
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")

    embed_dim = 1024
    batch_size = 1
    num_workers = 8
    epochs = 20
    label_dict = {"LUAD": 0, "LUSC": 1}


class H5Dataset(Dataset):

    def __init__(self, h5_path, filter_mapping, s_to_p_mapping=None,
                 mask_pkl=None, mask_tile_category=None, mask_tile_threshold=None, invert_threshold=None):
        self.h5_path = h5_path
        self.h5_paths = os.listdir(h5_path)
        self.h5_paths = [h5_path / path for path in self.h5_paths if int(path[:-3]) in filter_mapping.keys()]
        self.h5_labels = [filter_mapping[int(h5_path.name[:-3])] for h5_path in self.h5_paths]
        if s_to_p_mapping is None:
            self.s_to_p_mapping = None
        else:
            self.s_to_p_mapping = s_to_p_mapping
        if mask_pkl is None:
            self.mask_pkl = None
        else:
            self.mask_pkl = mask_pkl
            self.mask_threshold = mask_tile_threshold
            self.mask_tile_category = mask_tile_category
            self.comp_func = operator.gt if invert_threshold else operator.lt
            self.invert_threshold = invert_threshold

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        assets, attrs = self.read_assets_from_h5(self.h5_paths[idx], self.mask_pkl)
        # print("after", assets["tile_embeds"].shape)
        assets["labels"] = self.h5_labels[idx]
        if self.s_to_p_mapping is None:
            assets["patient_int"] = int(self.h5_paths[idx].name[:-3])
        else:
            assets["patient"] = self.s_to_p_mapping[int(self.h5_paths[idx].name[:-3])]
        return assets

    def read_assets_from_h5(self, h5_path: str, mask_pkl=None) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            if mask_pkl is not None:
                if self.mask_tile_category == "rand":
                    mask_bool = self.comp_func(np.random.rand(f["tile_embeds"].shape[0]), self.mask_threshold)
                elif self.mask_tile_category == "in_slide":
                    in_slide_threshold = np.quantile(mask_pkl[h5_path.stem], self.mask_threshold)
                    mask_bool = self.comp_func(mask_pkl[h5_path.stem], in_slide_threshold)
                elif self.mask_tile_category == "in_slide_weight":
                    slide_ambiguity = mask_pkl[h5_path.stem]
                    # Normalize the ambiguity scores to [0, 1]
                    # slide_ambiguity = (slide_ambiguity - slide_ambiguity.min()) / (slide_ambiguity.max() - slide_ambiguity.min())
                    # Calculate the selection probability inversely proportional to the ambiguity scores
                    selection_probabilities = 1 - slide_ambiguity
                    selection_probabilities /= np.sum(selection_probabilities)
                    mask_bool = np.zeros(len(selection_probabilities), dtype=bool)
                    if self.invert_threshold:
                        num_selected = int((1 - self.mask_threshold) * len(selection_probabilities))
                    else:
                        num_selected = int(self.mask_threshold * len(selection_probabilities))
                    # print("selection_probabilities", selection_probabilities)
                    selected_indices = np.random.choice(len(selection_probabilities), num_selected, replace=False,
                                                        p=selection_probabilities)
                    mask_bool[selected_indices] = True
                    mask_bool = ~mask_bool if self.invert_threshold else mask_bool
                else:
                    mask_bool = self.comp_func(mask_pkl[h5_path.stem], self.mask_threshold)
            for key in f.keys():
                if mask_pkl is not None:
                    assets[key] = f[key][:][mask_bool]
                else:
                    assets[key] = f[key][:]
                # print("before", mask_bool.shape, "after", mask_bool.sum())
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs


def train(config, args, i, j):
    if args.spec_norm_bound is None:
        middle_prefix = f"{config.model_name}"
    else:
        middle_prefix = f"{config.model_name}_sn{args.spec_norm_bound}_hyperopt"
    if hasattr(args, "force_middle_prefix") and args.force_middle_prefix is not None:
        middle_prefix = args.force_middle_prefix
        print("force_middle_prefix", middle_prefix)
    args.to_destination = args.save_destination / middle_prefix / f"t{i}f{j}"
    df = pd.read_csv(config.dataset_csv)
    external_df = pd.read_csv(config.external_dataset_csv)
    df["cohort"] = df["cohort"].apply(lambda x: config.label_dict[x])
    external_df["cohort"] = external_df["cohort"].apply(lambda x: config.label_dict[x])
    split_key = f"tao_split_trial_{i}_fold{j}"
    train_df, val_df, test_df = df[df[split_key] == "train"], df[df[split_key] == "val"], df[df[split_key] == "test"]
    if args.shrink_etest:
        external_df = external_df[external_df[split_key] == "test"]
    # create a dictionary mapping slide names to cohort
    train_df_mapping = train_df.set_index("slide_int")["cohort"].to_dict()
    val_df_mapping = val_df.set_index("slide_int")["cohort"].to_dict()
    test_df_mapping = test_df.set_index("slide_int")["cohort"].to_dict()
    external_df_mapping = external_df.set_index("slide_int")["cohort"].to_dict()
    if args.mask_tile:
        if not args.data_specific_evaluation:
            if args.shrink_etest:
                itest_mask_pkl = None
            else:
                itest_mask_pkl = load_pickle_model(config.mask_func(args.model_name, "itest", args.tuning_method))[
                    f"t{i}f{j}"]
            etest_mask_pkl = load_pickle_model(config.mask_func(args.model_name, "etest", args.tuning_method))[f"t{i}f{j}"]
        else:
            # each provide with their own best tuning_method
            itest_mask_pkl = load_pickle_model(config.mask_func(args.model_name, "itest", 0))[f"t{i}f{j}"]
            etest_mask_pkl = load_pickle_model(config.mask_func(args.model_name, "etest", 2))[f"t{i}f{j}"]


        if args.mask_tile_category == "rand" or args.mask_tile_category == "in_slide" or args.mask_tile_category == "in_slide_weight":  # float
            mask_tile_threshold = args.mask_tile_threshold
            val_mask_tile_threshold = args.mask_tile_threshold
        else:
            if args.shrink_etest:
                mask_tile_threshold = np.inf
            else:
                mask_tile_threshold = itest_mask_pkl["train_quantile_list"][args.mask_tile_threshold]
            val_mask_tile_threshold = itest_mask_pkl["val_quantile_list"][args.mask_tile_threshold]
        if not args.shrink_etest:
            print("Train Quantile", itest_mask_pkl["train_quantile_list"])
            print("Test Quantile", itest_mask_pkl["val_quantile_list"])
        print("Removing Train tiles with quantile larger than", mask_tile_threshold)
        print("Removing Test tiles with quantile larger than", val_mask_tile_threshold)
    else:
        itest_mask_pkl = None
        etest_mask_pkl = None
        mask_tile_threshold = None
        val_mask_tile_threshold = None

    def create_slide_int_to_patient_mapping(df):
        slide_int_to_patient = dict(zip(df["slide_int"], df["patient"]))
        return slide_int_to_patient

    i_s_to_p_mapping = create_slide_int_to_patient_mapping(df)
    e_s_to_p_mapping = create_slide_int_to_patient_mapping(external_df)

    if args.model_name == "inception-sngp":
        config.dataset_h5 = config.persistant_h5 / f"t{i}f{j}"
        config.etest_h5 = config.persistant_h5 / f"t{i}f{j}_cptac"

    train_dataset, val_dataset, test_dataset, etest_dataset = (H5Dataset(config.dataset_h5, train_df_mapping,
                                                                         s_to_p_mapping=i_s_to_p_mapping,
                                                                         mask_pkl=itest_mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold),
                                                               H5Dataset(config.dataset_h5, val_df_mapping,
                                                                         s_to_p_mapping=i_s_to_p_mapping,
                                                                         mask_pkl=itest_mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold),
                                                               H5Dataset(config.dataset_h5, test_df_mapping,
                                                                         s_to_p_mapping=i_s_to_p_mapping,
                                                                         mask_pkl=itest_mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold),
                                                               H5Dataset(config.etest_h5, external_df_mapping,
                                                                         s_to_p_mapping=e_s_to_p_mapping,
                                                                         mask_pkl=etest_mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold))
    print(f"train dataset size: {len(train_dataset)}, val dataset size: {len(val_dataset)}, "
          f"test dataset size: {len(test_dataset)}, etest dataset size: {len(etest_dataset)}")

    def collate_fn(batch):
        return {
            'tile_embeds': torch.stack([torch.from_numpy(x['tile_embeds']) for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch]),
            "patient": np.array([x['patient'] for x in batch])
        }

    train_loader, val_loader, test_loader, etest_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                                                     num_workers=config.num_workers,
                                                                     collate_fn=collate_fn), \
        DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn), \
        DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn), \
        DataLoader(etest_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn,
                   shuffle=False)
    model = GatedABMIL(embed_dim=config.embed_dim).cuda()
    if args.spec_norm_bound is not None:
        model = convert_to_sn_my(model, args.spec_norm_replace_list, args.spec_norm_bound)
    if args.gaussian_process:
        GP_KWARGS = GP_KWARGS_CONCH if args.model_name == "conch" else GP_KWARGS_UNI
        if args.gp_num_inducing is not None:
            assert args.gp_num_inducing > 512, "Number of inducing points should be larger than 512"
        GP_KWARGS["num_inducing"] = args.gp_num_inducing if args.gp_num_inducing is not None else GP_KWARGS["num_inducing"]
        if args.hyperopt:
            for key, value in args.memorized_hyperopt.items():
                GP_KWARGS[key] = value
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    print("parameter", sum(p.numel() for p in model.parameters()))
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    total_iter = len(train_dataset) // config.batch_size * config.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0
    if args.evaluate_only:
        print("args.to_destination", args.to_destination)
        model.load_state_dict(torch.load(args.to_destination / "best_model.pth"))
        if not args.data_specific_evaluation:
            if not args.shrink_etest:  # when shrink etest, we do not need evaluate on the itest
                itest_acc, itest_bacc, itest_auc = evaluate(model, test_loader, args, i, j, "itest")
            else:
                itest_acc, itest_bacc, itest_auc = 0, 0, 0
        else:
            # in data_specific_evaluation, we evaluate the itest and the etest for any other reason
            # note that the shrink_etest must be True, and it will overwrite the shrink_etest
            itest_acc, itest_bacc, itest_auc = evaluate(model, test_loader, args, i, j, "itest")
        etest_acc, etest_bacc, etest_auc = evaluate(model, etest_loader, args, i, j, "etest")
        print(
            f"Trial {i} Fold {j} Test Accuracy: {itest_acc}, Test Balanced Accuracy: {itest_bacc}, Test AUC: {itest_auc}")
        print(
            f"Trial {i} Fold {j} External Test Accuracy: {etest_acc}, External Test Balanced Accuracy: {etest_bacc}, External Test AUC: {etest_auc}")
        return itest_acc, itest_bacc, itest_auc, etest_acc, etest_bacc, etest_auc
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        if args.gaussian_process:
            # GP_KWARGS["gp_cov_discount_factor"] == -1, in fact, it is not necessary when momentum != -1
            model.classifier.reset_covariance_matrix()
            kwargs = {'return_random_features': False, 'return_covariance': False,
                      'update_precision_matrix': True, 'update_covariance_matrix': False}
        else:
            kwargs = {}
        for idx, assets in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            H = assets['tile_embeds'].float().cuda()
            labels = assets['labels'].long().cuda()
            preds = model(H, **kwargs)
            loss = criterion(preds, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"{i * 4 + j} Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(train_loader)}")
        val_accuracy, _, _ = evaluate(model, val_loader, args, i, j)
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(args.to_destination, exist_ok=True)
            torch.save(model.state_dict(), args.to_destination / "best_model.pth")

    model.load_state_dict(torch.load(args.to_destination / f"best_model.pth"))
    val_accuracy, _, _ = evaluate(model, val_loader, args, i, j, "ival")
    itest_acc, itest_bacc, itest_auc = evaluate(model, test_loader, args, i, j, "itest")
    etest_acc, etest_bacc, etest_auc = evaluate(model, etest_loader, args, i, j, "etest")
    print(f"Trial {i} Fold {j} Test Accuracy: {itest_acc}, Test Balanced Accuracy: {itest_bacc}, Test AUC: {itest_auc}")
    print(
        f"Trial {i} Fold {j} External Test Accuracy: {etest_acc}, External Test Balanced Accuracy: {etest_bacc}, External Test AUC: {etest_auc}")
    return itest_acc, itest_bacc, itest_auc, etest_acc, etest_bacc, etest_auc


def evaluate(model, loader, args, i, j, tag=None):
    model.eval()
    labels = []
    patient_id = []
    logits_list = []
    uncertainty_list = []
    if args.gaussian_process:
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}
    else:
        eval_kwargs = {}
    for idx, assets in tqdm(enumerate(loader)):
        H = assets['tile_embeds'].float().cuda()
        output = model(H, **eval_kwargs)
        if isinstance(output, tuple):
            logits, covariance = output
            logits = logits.cpu().detach().numpy()
            uncertainty = torch.diagonal(covariance).cpu().detach().numpy()
            uncertainty_list.extend(uncertainty)
        else:
            logits = output.cpu().detach().numpy()
        logits_list.extend(logits)
        labels.extend(assets['labels'].long())
        patient_id.extend(assets['patient'])
    logits = np.stack(logits_list, axis=0)
    labels = np.stack(labels, axis=0)
    df = pd.DataFrame({
        'patient_id': np.array(patient_id),
        'logit_0': logits[:, 0],
        'logit_1': logits[:, 1],
        'labels': labels,
    })

    agg_dict = {
        'logit_0': 'mean',
        'logit_1': 'mean',
        'labels': 'first',
    }

    if uncertainty_list:
        df['uncertainty'] = uncertainty_list
        agg_dict['uncertainty'] = 'mean'

    agg_df = df.groupby('patient_id').agg(agg_dict).reset_index()

    results = {
        'logit': np.array(agg_df[['logit_0', 'logit_1']]),
        'label': np.array(agg_df['labels']),
        'prob': softmax(np.array(agg_df[['logit_0', 'logit_1']]), axis=1),
        'patient': np.array(agg_df['patient_id'])
    }

    if uncertainty_list:
        results['uncertainty'] = np.array(agg_df['uncertainty'])

    print(f"agg from {len(df)} to {len(agg_df)}")
    try:
        acc = accuracy_score(results['label'], results['prob'].argmax(axis=1))
        bacc = balanced_accuracy_score(results['label'], results['prob'].argmax(axis=1))
        auc = roc_auc_score(results['label'], results['prob'][:, 1])
    except Exception as e:
        print(f"Error: {e}")
        acc, bacc, auc = 0, 0, 0
    if args.save_to_parquet and tag is not None:
        save_df = pd.DataFrame({
            'Outcome 0-y_pred0': results['logit'][:, 0],
            'Outcome 0-y_pred1': results['logit'][:, 1],
            'Outcome 0-y_true': results['label'],
            'prob_0': results['prob'][:, 0],
            'prob_1': results['prob'][:, 1],
            'patient': results['patient']
        })
        if uncertainty_list:
            save_df['Outcome 0-uncertainty0'] = results['uncertainty']
        if args.mask_tile:
            save_df_name = f"patient_predictions_{tag}_t{i}f{j}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}_tuning{args.tuning_method}.parquet.gzip"
        else:
            save_df_name = f"patient_predictions_{tag}_t{i}f{j}.parquet.gzip"
        save_df.to_parquet(args.to_destination / save_df_name, compression="gzip")

    return acc, bacc, auc


def load_training_config(model_name):
    if model_name == "uni":
        return UniTrainingConfig
    elif model_name == "conch":
        return ConchTrainingConfig
    elif model_name == "inception-sngp":
        return InceptionTrainingConfig
    else:
        raise ValueError(f"Invalid model name {model_name}")

def main_spawn(args):
    training_config = load_training_config(model_name=args.model_name)
    start_time = time.time()
    for config in [training_config]:
        print(config.model_name)
        metric_dict = defaultdict(list)
        for i in range(args.trial):
            for j in range(args.fold):
                itest_acc, itest_bacc, itest_auc, etest_acc, etest_bacc, etest_auc = train(config, args, i, j)
                metric_dict['itest_acc'].append(itest_acc)
                metric_dict['itest_bacc'].append(itest_bacc)
                metric_dict['itest_auc'].append(itest_auc)
                metric_dict['etest_acc'].append(etest_acc)
                metric_dict['etest_bacc'].append(etest_bacc)
                metric_dict['etest_auc'].append(etest_auc)

        df_metrics = pd.DataFrame(metric_dict)
        summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
        summary_metrics.columns = ['mean', 'std']
        summary_metrics["metrics"] = ["itest_acc", "itest_bacc", "itest_auc", "etest_acc", "etest_bacc",
                                      "etest_auc"]
        summary_metrics["mean-std"] = (summary_metrics["mean"].round(4).astype(str) + "+-" +
                                       summary_metrics["std"].round(4).astype(str))
        summary_metrics.drop(columns=["mean", "std"], inplace=True)
        summary_metrics['tag'] = f"{args.model_name}_sn{args.spec_norm_bound}_gp{int(args.gaussian_process)}"
        summary_metrics = summary_metrics.pivot(index='tag', columns='metrics', values='mean-std').reset_index()
        for metric in metric_dict.keys():
            summary_metrics[metric + "_list"] = [metric_dict[metric]]  # additional original value for each metric
        if args.mask_tile is False:
            mask_ratio = "no_mask"
        else:
            mask_ratio = f"mask_category{args.mask_tile_category}_thres{args.mask_tile_threshold}"
        summary_metrics["mask_ratio"] = mask_ratio
        if args.hyperopt:
            summary_metrics["hyperopt"] = str(args.memorized_hyperopt)
        print(summary_metrics)

        results_file_path = args.save_destination / args.results_file_path
        if results_file_path.exists():
            existing_data = pd.read_csv(results_file_path)
            updated_data = pd.concat([existing_data, summary_metrics],
                                     ignore_index=True)  # Concatenating with ignore_index=True
            updated_data.to_csv(results_file_path, index=False)
        else:
            summary_metrics.to_csv(results_file_path, index=False)
        for key in metric_dict:
            print(f"{key}: {np.mean(metric_dict[key])}+-{np.std(metric_dict[key])}, runs {len(metric_dict[key])}")
        torch.cuda.empty_cache()
    print("Duration: (s)", time.time() - start_time, "minutes: (m)", (time.time() - start_time) / 60)
    if args.hyperopt:
        wandb.summary["itest_acc"] = np.mean(metric_dict["itest_acc"])
        wandb.summary["etest_acc"] = np.mean(metric_dict["etest_acc"])



def hyperopt_loop(args):
    """
        GP_KWARGS["gp_input_normalization"] = args.memorized_hyperopt["gp_input_normalization"]
        GP_KWARGS["gp_scale"] = args.memorized_hyperopt["gp_scale"]
        GP_KWARGS["gp_output_imagenet_initializer"] = args.memorized_hyperopt["gp_output_imagenet_initializer"]
        GP_KWARGS["num_inducing"] = args.memorized_hyperopt["num_inducing"]
    """


    run = wandb.init()
    args_hyperopt = copy.deepcopy(args)
    args_hyperopt.memorized_hyperopt = run.config
    args_hyperopt.force_middle_prefix = f"{args.model_name}_sn{args.spec_norm_bound}_hyperopt"
    for key, value in args_hyperopt.memorized_hyperopt.items():
        # transfer gp_input_normalization to gin, gp_scale to gsc, gp_output_imagenet_initializer to gpoi
        split_list = key.split("_")
        new_key = [split_key[0] for split_key in split_list]
        new_key = "".join(new_key)
        if isinstance(value, bool):
            value = int(value)
        args_hyperopt.force_middle_prefix += f"_{new_key}{value}"
    # args_hyperopt.force_middle_prefix = (f"{args.model_name}_sn{args.spec_norm_bound}_hyperopt_gin{int(args_hyperopt.memorized_hyperopt['gp_input_normalization'])}_"
    #                             f"gsc{args_hyperopt.memorized_hyperopt['gp_scale']}_gpoi{int(args_hyperopt.memorized_hyperopt['gp_output_imagenet_initializer'])}_"
    #                             f"gnind{args_hyperopt.memorized_hyperopt['num_inducing']}")
    print("args_hyperopt.memorized_hyperopt", args_hyperopt.memorized_hyperopt)
    main_spawn(args_hyperopt)
    wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    _, args = get_abmil_params()
    print(args)
    seed_everything(args.seed)
    if not args.hyperopt:
        main_spawn(args)
    else:

        sweep_configuration = {
            "method": "grid",
            "name": f"sweep_abmil_{args.model_name}",
            "metric": {
                "goal": "maximize",
                "name": "itest_acc",
            },
            "parameters": {# linear kernel does not need inducing points, gp_scale
                "gp_kernel_type": {"values": ["linear"]},
                "gp_input_normalization": {"values": [True, False]},
                "gp_output_imagenet_initializer": {"values": [True, False]},
            },
            # "parameters": {
            #     "gp_kernel_type": {"values": ["gaussian"]},
            #     "gp_input_normalization": {"values": [True, False]},
            #     "gp_scale": {"values": [1., 2., 3., 1.5, 2.5]},
            #     "num_inducing": {"values": [256, 512,]}, # [1024, 2048]
            #     "gp_output_imagenet_initializer": {"values": [True, False]},
            # },
        }
        project_name = f"abmil_{args.model_name}_sn{args.spec_norm_bound}_linear"
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
        wandb.agent(sweep_id, function=partial(hyperopt_loop, args=args))


