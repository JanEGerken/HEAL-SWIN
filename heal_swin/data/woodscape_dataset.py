import os
import sys

import numpy as np
from torch.utils.data import Dataset


class WoodscapeDataset(Dataset):
    def __init__(
        self,
        cam_pos=None,
        train_share=0.8,
        part="train",
        padding=[0, 0, 0, 0],
        shuffle_train_val_split=True,
        woodscape_version=None,
        training_data_fraction=1.0,
        data_fraction_seed=42,
    ):
        super().__init__()
        self.woodscape_version = woodscape_version
        self.paths_dict = self.get_paths()

        if cam_pos is None:
            self.paths = []
            for paths in self.paths_dict.values():
                self.paths += paths
        elif cam_pos in self.paths_dict.keys():
            self.paths = self.paths_dict[cam_pos]
        else:
            print(f"camera position {cam_pos} unknown", file=sys.stderr)
            sys.exit(1)

        self.shuffle_paths(shuffle_train_val_split)

        assert 0 <= train_share <= 1
        if part == "train":
            min_idx = 0
            max_idx = int(np.floor(len(self.paths) * train_share))
        elif part == "val":
            min_idx = int(np.ceil(len(self.paths) * train_share))
            max_idx = len(self.paths) - 1
        else:
            print(f"Unknown part {part}, set to 'train' or 'val'", file=sys.stderr)
            sys.exit(1)
        self.paths = self.paths[min_idx : max_idx + 1]
        if part == "train":
            self.take_subset_of_paths(training_data_fraction, data_fraction_seed)

        self.update_paths()

    def get_paths(self):
        root_dir = self.get_dir()

        if not os.path.isdir(root_dir):
            print("Could not find directory at", root_dir, file=sys.stderr)
            sys.exit(1)

        ext = self.get_extension()
        ext_in_root_dir = [
            entry.path for entry in os.scandir(root_dir) if ext in entry.name
        ]
        fv = [fv_img for fv_img in ext_in_root_dir if "FV" in fv_img]
        rv = [rv_img for rv_img in ext_in_root_dir if "RV" in rv_img]
        mvl = [mvl_img for mvl_img in ext_in_root_dir if "MVL" in mvl_img]
        mvr = [mvr_img for mvr_img in ext_in_root_dir if "MVR" in mvr_img]

        result = {"fv": fv, "rv": rv, "mvl": mvl, "mvr": mvr}

        return result

    def get_extension(self):
        """Overwrite in child with function which returns the extension of the desired data"""
        return ".png"

    def get_dir(self):
        """Overwrite in child with function that returns the directory of the desired data"""
        pass

    def shuffle_paths(self, shuffle):
        """Shuffle the paths to get a good mix of training and validation data. In order to avoid
        overlap, it's crucial to fix the random seed and sort the paths before shuffling, since the
        train and test datasets will be created by different instances of this class.

        """
        self.paths.sort()
        if shuffle:
            r = np.random.RandomState(42)
            idcs = r.permutation(len(self.paths))
            self.paths = np.array(self.paths)[idcs]

    def take_subset_of_paths(self, fraction, seed):
        """Take a `fraction` of the paths for the training data. This has been sorted and shuffled before.
        Different subsets can be obtained using different `seed`s.

        Should be used only for training data.
        """
        r = np.random.RandomState(seed)
        subset_len = np.ceil(len(self.paths) * fraction).astype(int)
        self.paths = self.paths[r.permutation(len(self.paths))][0:subset_len]

    def update_paths(self):
        """Call this function after paths have been set to new values"""
        for path in self.paths:
            if not os.path.isfile(path):
                print(f"Could not find file {path}", file=sys.stderr)
                sys.exit(1)
        self.file_names = np.array(
            [os.path.basename(file_name) for file_name in self.paths]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """Overwrite in child with function which returns data from self.paths[idx] as pytorch
        tensor"""
        pass
