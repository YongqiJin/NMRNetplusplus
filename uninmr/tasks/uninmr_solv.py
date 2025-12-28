# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from scipy import stats
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    LMDBDataset,
    RawLabelDataset,
)
from uninmr.data import (
    KeyDataset,
    SolventDataset,
    ConstantDataset,
    ConformerSampleDataset,
    TTADataset,
    IndexDataset,
    TTAIndexDataset,
    ToTorchDataset,
    MaskPointsDataset,
    DistanceDataset,
    GlobalDistanceDataset,
    EdgeTypeDataset,
    RightPadDataset3D,
    PrependAndAppend2DDataset,
    PrependAndAppend3DDataset,
    RightPadDataset2D0,
    LatticeNormalizeDataset,
    LatticeMatrixNormalizeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    TargetScalerDataset,
    FoldLMDBDataset,
    StackedLMDBDataset,
    SplitLMDBDataset,
    SelectTokenDataset,
    FilterDataset,
    MergedDataset,
)
from unicore.tasks import UnicoreTask, register_task
from uninmr.utils import parse_select_atom, TargetScaler
logger = logging.getLogger(__name__)

@register_task("uninmr_solv")
class UniNMRTaskwithSolvent(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--saved-dir",
            help="saved dir"
        )
        parser.add_argument(
            "--classification-head-name",
            default="nmr_head",
            help="finetune downstream task name"
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers"
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--has-matid",
            action="store_true",
            help="whether already has matid",
        )
        parser.add_argument(
            "--conformer-augmentation",
            action="store_true",
            help="using conformer augmentation",
        )
        parser.add_argument(
            "--conf-size",
            type=int,
            default=10,
            help="conformer nums per structure",
        )
        parser.add_argument(
            "--global-distance",
            action="store_true",
            help="use global distance",
        )  
        parser.add_argument(
            "--atom-descriptor",
            type=int,
            default=0,
            help="use extra atom descriptor",
        )  
        parser.add_argument(
            "--selected-atom",
            default="All",
            help="select atom: All or H or H&C&F...",
        )  
        parser.add_argument(
            '--split-mode', 
            type=str, 
            default='predefine',
            choices=['predefine', 'cross_valid', 'random', 'infer'],
        )
        parser.add_argument(
            "--nfolds",
            default=5,
            type=int,
            help="cross validation split folds"
        )
        parser.add_argument(
            "--fold",
            default=0,
            type=int,
            help='local fold used as validation set, and other folds will be used as train set'
        )
        parser.add_argument(
            "--cv-seed",
            default=42,
            type=int,
            help="random seed used to do cross validation splits"
        )
        parser.add_argument(
            "--unlabeled-data",
            default=None,
            type=str,
            help="path to unlabeled data for semi-supervised learning",
        )
        parser.add_argument(
            "--unlabeled-weight",
            default=1.0,
            type=float,
            help="weight for unsupervised loss",
        )
        parser.add_argument(
            "--ratios",
            default=[1,0],
            type=float,
            nargs='+',  # 支持输入一个或多个值
            help="ratios for labeled and unlabeled data",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.selected_token = parse_select_atom(self.dictionary, args.selected_atom)
        if args.saved_dir == None:
            self.args.saved_dir = args.save_dir
        self.target_scaler = TargetScaler(args.saved_dir)
        self.args.ratios = [i / sum(args.ratios) for i in args.ratios]
        self.train_residues = torch.empty(0, dtype=torch.float32)
        self.valid_residues = torch.empty(0, dtype=torch.float32)
        self.train_unlabeled_losses = torch.empty(0, dtype=torch.float32)
        self.valid_unlabeled_losses = torch.empty(0, dtype=torch.float32)
        self.weights = torch.empty(0, dtype=torch.float32)

        if self.args.split_mode =='predefine':
            train_path = os.path.join(self.args.data, "train" + ".lmdb")
            self.train_dataset = LMDBDataset(train_path)
            token_dataset = KeyDataset(self.train_dataset, "atoms")
            token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            atoms_target_mask_dataset = KeyDataset(self.train_dataset, "atoms_target_mask")
            select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
            filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]
            self.train_dataset = FilterDataset(self.train_dataset, filter_list)

            valid_path = os.path.join(self.args.data, "valid" + ".lmdb")
            self.valid_dataset = LMDBDataset(valid_path)
            token_dataset = KeyDataset(self.valid_dataset, "atoms")
            token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            atoms_target_mask_dataset = KeyDataset(self.valid_dataset, "atoms_target_mask")
            select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
            filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]
            self.valid_dataset = FilterDataset(self.valid_dataset, filter_list)

            atoms_target = np.concatenate([np.array(self.train_dataset[i]['atoms_target']) for i in range(len(self.train_dataset))], axis=0) 
            atoms_target_mask = np.concatenate([np.array(self.train_dataset[i]['atoms_target_mask']) for i in range(len(self.train_dataset))], axis=0) 
            self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=self.args.save_dir)
        elif self.args.split_mode == 'infer':
            valid_dataset = self.__init_data_infer(self.args.data)
            if self.args.unlabeled_data:
                unlabeled_dataset = self.__init_data_infer(self.args.unlabeled_data)
                self.valid_dataset = MergedDataset([valid_dataset, unlabeled_dataset], [1,0], mode="append")
            else:
                self.valid_dataset = valid_dataset
        else:
            self.__init_data()
            if self.args.unlabeled_data:
                self.__init_unlabeled_data()
                
    def __init_data_infer(self, data):
        valid_path = os.path.join(data, "valid" + ".lmdb")
        self.valid_dataset = LMDBDataset(valid_path)
        token_dataset = KeyDataset(self.valid_dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(self.valid_dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
        solvent_dataset = SolventDataset(self.valid_dataset, "nmr_solvent")
        filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]
        valid_dataset = FilterDataset(self.valid_dataset, filter_list)
        return valid_dataset

    def __init_data(self):
        data_path = os.path.join(self.args.data, 'train.lmdb')
        raw_dataset = LMDBDataset(data_path)
        token_dataset = KeyDataset(raw_dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(raw_dataset, "atoms_target_mask")
        solvent_dataset = SolventDataset(raw_dataset, "nmr_solvent")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
        filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]
        raw_dataset = FilterDataset(raw_dataset, filter_list)

        atoms_target = np.concatenate([np.array(raw_dataset[i]['atoms_target']) for i in range(len(raw_dataset))], axis=0) 
        atoms_target_mask = np.concatenate([np.array(raw_dataset[i]['atoms_target_mask']) for i in range(len(raw_dataset))], axis=0) 

        if self.args.split_mode == 'cross_valid':
            train_folds = []
            for _fold in range(self.args.nfolds):
                if _fold == 0:
                    parent_dir = os.path.dirname(self.args.saved_dir)
                    self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=parent_dir)
                    cache_fold_info = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
                if _fold == self.args.fold:
                    self.valid_dataset = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
                if _fold != self.args.fold:
                    train_folds.append(FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
            self.train_dataset = StackedLMDBDataset(train_folds)
        elif self.args.split_mode == 'random':
            self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=self.args.saved_dir)
            cache_fold_info = SplitLMDBDataset(raw_dataset, self.args.seed, 0).get_fold_info()   
            self.train_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 0, cache_fold_info=cache_fold_info)
            self.valid_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 1, cache_fold_info=cache_fold_info)

    def __init_unlabeled_data(self):
        unlabeled_data_path = os.path.join(self.args.unlabeled_data, "train" + ".lmdb")
        unlabeled_raw_dataset = LMDBDataset(unlabeled_data_path)
        token_dataset = KeyDataset(unlabeled_raw_dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(unlabeled_raw_dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
        solvent_dataset = KeyDataset(unlabeled_raw_dataset, "nmr_solvent")
        filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]
        unlabeled_raw_dataset = FilterDataset(unlabeled_raw_dataset, filter_list)
        
        if self.args.split_mode == 'cross_valid':
            unlabeled_train_folds = []
            for _fold in range(self.args.nfolds):
                if _fold == 0:
                    # parent_dir = os.path.dirname(self.args.saved_dir)
                    # self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=parent_dir)
                    cache_fold_info = FoldLMDBDataset(unlabeled_raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
                if _fold == self.args.fold:
                    self.unlabeled_valid_dataset = FoldLMDBDataset(unlabeled_raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
                if _fold != self.args.fold:
                    unlabeled_train_folds.append(FoldLMDBDataset(unlabeled_raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
            self.unlabeled_train_dataset = StackedLMDBDataset(unlabeled_train_folds)
        elif self.args.split_mode == 'random':
            self.unlabeled_train_dataset = SplitLMDBDataset(unlabeled_raw_dataset, self.args.seed, 0, cache_fold_info=cache_fold_info)
            self.unlabeled_valid_dataset = SplitLMDBDataset(unlabeled_raw_dataset, self.args.seed, 1, cache_fold_info=cache_fold_info)
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """

        labeled_nest_dataset = self.processs_dataset(split, "labeled")
        if self.args.unlabeled_data:
            unlabeled_nest_dataset = self.processs_dataset(split, "unlabeled")
        
        if split in ["train", "train.small"] and self.args.unlabeled_data:
            nest_dataset = MergedDataset([labeled_nest_dataset, unlabeled_nest_dataset], self.args.ratios, mode="repeat")
            logging.info("Number of Train Dataset: labeled={}, unlabeled={}, total={}".format(len(labeled_nest_dataset), len(unlabeled_nest_dataset), len(nest_dataset)))
        elif split in ["valid"] and self.args.unlabeled_data:
            nest_dataset = MergedDataset([labeled_nest_dataset, unlabeled_nest_dataset], [1,0], mode="append")
            logging.info("Number of Valid Dataset: labeled={}, unlabeled={}, total={}".format(len(labeled_nest_dataset), len(unlabeled_nest_dataset), len(nest_dataset)))
        else:
            nest_dataset = labeled_nest_dataset

        self.datasets[split] = nest_dataset

    def processs_dataset(self, split, data_name, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        assert data_name in ["labeled", "unlabeled"]
        
        if split == 'train':
            dataset = self.train_dataset if data_name == "labeled" else self.unlabeled_train_dataset
        elif split == 'valid':
            dataset = self.valid_dataset if data_name == "labeled" else self.unlabeled_valid_dataset

        if data_name == "labeled":
            is_labeled_dataset = ConstantDataset(dataset, 1)
        else:
            is_labeled_dataset = ConstantDataset(dataset, 0)

        if self.args.has_matid:
            matid_dataset = KeyDataset(dataset, "matid")
        else:
            matid_dataset = IndexDataset(dataset)

        if self.args.conformer_augmentation:
            if split == 'train':
                dataset = ConformerSampleDataset(dataset, self.seed, "atoms", "coordinates_list")
            else:
                dataset = TTADataset(dataset, self.seed, "atoms", "coordinates_list", self.args.conf_size)
                matid_dataset = TTAIndexDataset(matid_dataset, self.args.conf_size)
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        dataset = NormalizeDataset(dataset, "coordinates")

        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
        solvent_dataset = SolventDataset(dataset, "nmr_solvent")

        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos())
        select_atom_dataset = PrependAndAppend(select_atom_dataset, self.dictionary.pad(), self.dictionary.pad())
        
        coord_dataset = ToTorchDataset(coord_dataset, 'float32')

        if self.args.global_distance:
            lattice_matrix_dataset = LatticeMatrixNormalizeDataset(dataset, 'lattice_matrix')
            logger.info("use global distance: {}".format(self.args.global_distance))
            distance_dataset = GlobalDistanceDataset(coord_dataset, lattice_matrix_dataset)
            distance_dataset = PrependAndAppend3DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset3D(distance_dataset, pad_idx=0)
        else:
            distance_dataset = DistanceDataset(coord_dataset)
            distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        edge_type = EdgeTypeDataset(token_dataset, len(self.dictionary))

        tgt_dataset = KeyDataset(dataset, "atoms_target")
        tgt_dataset = TargetScalerDataset(tgt_dataset, self.target_scaler, self.args.num_classes)
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')

        tgt_dataset = PrependAndAppend(tgt_dataset, self.dictionary.pad(), self.dictionary.pad())

        if self.args.atom_descriptor != 0:
            atomdes_dataset = KeyDataset(dataset, "atoms_descriptor")
            atomdes_dataset = ToTorchDataset(atomdes_dataset, dtype='float32')
            atomdes_dataset = PrependAndAppend(atomdes_dataset, self.dictionary.pad(), self.dictionary.pad())
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                            "atom_descriptor": RightPadDataset2D0(
                                atomdes_dataset,
                                pad_idx=0,
                            ),
                            "solvent": solvent_dataset,
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                    },
                )
        else:
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                            "solvent": solvent_dataset,
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                        "is_labeled": is_labeled_dataset,
                    },
                )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        return nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model.register_node_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
            extra_dim=self.args.atom_descriptor,
        )
        return model

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        self.valid_residues = torch.cat((self.valid_residues, logging_output['labeled_residues'].cpu()), dim=0)
        self.valid_unlabeled_losses = torch.cat((self.valid_unlabeled_losses, logging_output['unlabeled_losses'].cpu()), dim=0)
        return loss, sample_size, logging_output
    
    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        self.weights = torch.cat((self.weights, logging_output['weights'].cpu()), dim=0)
        self.train_residues = torch.cat((self.train_residues, logging_output['labeled_residues'].cpu()), dim=0)
        self.train_unlabeled_losses = torch.cat((self.train_unlabeled_losses, logging_output['unlabeled_losses'].cpu()), dim=0)
        return loss, sample_size, logging_output
    
    def begin_epoch(self, epoch, model):
        if epoch > 1:
            logging.info("===== train residues statistics =====")
            if len(self.train_residues) > 0:
                self.log_statistics(self.train_residues)
            logging.info("===== train unlabeled losses statistics =====")
            if len(self.train_unlabeled_losses) > 0:
                self.log_statistics(self.train_unlabeled_losses)
            logging.info("===== valid residues statistics =====")
            if len(self.valid_residues) > 0:
                self.log_statistics(self.valid_residues)
            logging.info("===== valid unlabeled losses statistics =====")
            if len(self.valid_unlabeled_losses) > 0:
                self.log_statistics(self.valid_unlabeled_losses)
            logging.info(f"Epoch {epoch-1} Train | Labeled MAE: {np.mean(np.abs(self.train_residues.numpy())) * self.target_scaler.scaler.scale_} "
                            + f"| Labeled RMSE: {np.sqrt(np.mean(self.train_residues.numpy() ** 2)) * self.target_scaler.scaler.scale_} "
                            + f"| Unlabeled MAE: {np.mean(self.train_unlabeled_losses.numpy()) * self.target_scaler.scaler.scale_} "
                            + f"| Unlabeled RMSE: {np.sqrt(np.mean(self.train_unlabeled_losses.numpy() ** 2)) * self.target_scaler.scaler.scale_} "
            )
            logging.info(f"Epoch {epoch-1} Valid | Labeled MAE: {np.mean(np.abs(self.valid_residues.numpy())) * self.target_scaler.scaler.scale_} "
                            + f"| Labeled RMSE: {np.sqrt(np.mean(self.valid_residues.numpy() ** 2)) * self.target_scaler.scaler.scale_} "
                            + f"| Unlabeled MAE: {np.mean(self.valid_unlabeled_losses.numpy()) * self.target_scaler.scaler.scale_} "
                            + f"| Unlabeled RMSE: {np.sqrt(np.mean(self.valid_unlabeled_losses.numpy() ** 2)) * self.target_scaler.scaler.scale_} "
            )
            if len(self.weights) > 0:
                logging.info("===== train weights statistics =====")
                self.log_statistics(self.weights)
            self.train_residues = torch.empty(0, dtype=torch.float32)
            self.valid_residues = torch.empty(0, dtype=torch.float32)
            self.train_unlabeled_losses = torch.empty(0, dtype=torch.float32)
            self.valid_unlabeled_losses = torch.empty(0, dtype=torch.float32)
            self.weights = torch.empty(0, dtype=torch.float32)

    def log_statistics(self, data):
        data = data.numpy()
        
        mae = np.mean(np.abs(data))
        rmse = np.sqrt(np.mean(data ** 2))
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        
        q = np.percentile(data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        logger.info(f"Statistics: num={len(data)}, mae={mae}, rmse={rmse}, min={min_val}, max={max_val}, median={median}, quantiles={q.tolist()}")
