import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.read import read_mesh


class CoMA(InMemoryDataset):
    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self,
                 root,
                 data_split,
                 split='interpolation',
                 test_exp='bareteeth',
                 transform=None,
                 pre_transform=None):
        """
        root: CoMA root folder (…/data/CoMA)
        data_split: "train", "val", or "test"
        """
        self.split = split
        self.test_exp = test_exp

        # Make sure processed/<split> exists
        if not osp.exists(osp.join(root, 'processed', self.split)):
            os.makedirs(osp.join(root, 'processed', self.split), exist_ok=True)

        super().__init__(root, transform, pre_transform)

        # After processing, load only the requested split
        if data_split == "train":
            path = self.processed_paths[0]
        elif data_split == "val":
            path = self.processed_paths[1]
        elif data_split == "test":
            path = self.processed_paths[2]
        else:
            raise RuntimeError('Expected data_split in ["train", "val", "test"].')

        # IMPORTANT: set _data / _slices, not just data/slices
        self._data, self._slices = torch.load(path)

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        # We manage raw files manually (raw/torus/*.ply)
        return []

    @property
    def processed_file_names(self):
        # We use three files under processed/interpolation/
        if self.split == 'interpolation':
            return [
                osp.join(self.split, 'training.pt'),
                osp.join(self.split, 'val.pt'),
                osp.join(self.split, 'test.pt'),
            ]
        else:
            raise RuntimeError(
                f"Expected split 'interpolation', got '{self.split}'. "
                "Extrapolation not implemented for torus dataset."
            )

    def download(self):
        # No download, everything is local
        pass

    # ------------------------------------------------------------------
    # Main processing logic
    # ------------------------------------------------------------------
    def process(self):
        print('Processing...')

        # ---- 1. Load labels (ids + age) ----
        torus_dir = osp.join(self.raw_dir, "torus")
        label_fp = osp.join(torus_dir, "labels.pt")
        print(f"[CoMA] Loading labels from: {label_fp}")
        labels_obj = torch.load(label_fp, map_location="cpu")

        if not isinstance(labels_obj, dict) or "ids" not in labels_obj or "age" not in labels_obj:
            raise RuntimeError(
                "labels.pt must be a dict with keys 'ids' and 'age', "
                f"but got keys: {list(labels_obj.keys())}"
            )

        ids_raw = labels_obj["ids"]
        age_raw = labels_obj["age"]

        def to_list(x):
            import numpy as np
            if torch.is_tensor(x):
                return x.cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x)

        ids_list = [str(s) for s in to_list(ids_raw)]
        age_list = to_list(age_raw)

        if len(ids_list) != len(age_list):
            raise RuntimeError(
                f"ids and age have different lengths: {len(ids_list)} vs {len(age_list)}"
            )

        # subject_id -> age
        label_map = {sid: float(age) for sid, age in zip(ids_list, age_list)}
        print(f"[CoMA] #subjects in labels: {len(label_map)}")

        # ---- 2. Collect .ply files ----
        fps = sorted(glob(osp.join(torus_dir, "*.ply")))
        if len(fps) == 0:
            raise RuntimeError(f"No .ply found in {torus_dir}")

        subjects_all = [osp.splitext(osp.basename(fp))[0] for fp in fps]
        print(f"[CoMA] #.ply files: {len(subjects_all)}")

        # Keep only subjects that have labels
        subjects = [s for s in subjects_all if s in label_map]
        print(f"[CoMA] #subjects with labels & meshes: {len(subjects)}")

        if len(subjects) == 0:
            raise RuntimeError(
                "No overlap between .ply basenames and labels['ids']. "
                "Check that ids in labels.pt match filenames (without .ply)."
            )

        # ---- 3. Train/val/test split on subject IDs ----
        X_train, X_temp = train_test_split(subjects, test_size=0.2, random_state=28)
        if len(X_temp) >= 2:
            X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=28)
        else:
            X_val, X_test = [], X_temp

        print(
            f"[CoMA] Split sizes -> train: {len(X_train)}, "
            f"val: {len(X_val)}, test: {len(X_test)}"
        )

        split_map = {sid: "train" for sid in X_train}
        split_map.update({sid: "val" for sid in X_val})
        split_map.update({sid: "test" for sid in X_test})

        train_val_test_files = {
            "train": X_train,
            "val": X_val,
            "test": X_test,
        }

        # ---- 4. Build PyG Data lists ----
        train_data_list, val_data_list, test_data_list = [], [], []

        for fp in tqdm(fps):
            subject = osp.splitext(osp.basename(fp))[0]
            split = split_map.get(subject, None)
            if split is None:
                continue  # no label or not in splits

            data = read_mesh(fp)  # geometry only

            # Attach age label
            age_value = label_map[subject]
            data.y = torch.tensor([age_value], dtype=torch.float32)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if split == "train":
                train_data_list.append(data)
            elif split == "val":
                val_data_list.append(data)
            elif split == "test":
                test_data_list.append(data)

        print(
            f"[CoMA] Final counts -> train: {len(train_data_list)}, "
            f"val: {len(val_data_list)}, test: {len(test_data_list)}"
        )

        if len(train_data_list) == 0:
            raise RuntimeError("train_data_list is empty – something went wrong in splitting.")

        # ---- 5. Save in PyG format ----
        train_data = self.collate(train_data_list)  # (data, slices)
        val_data   = self.collate(val_data_list)
        test_data  = self.collate(test_data_list)

        torch.save(train_data, self.processed_paths[0])
        torch.save(val_data,   self.processed_paths[1])
        torch.save(test_data,  self.processed_paths[2])

        # extra info for later
        extra_proc_dir = osp.join(self.root, "processed")
        os.makedirs(extra_proc_dir, exist_ok=True)
        torch.save(
            train_val_test_files,
            osp.join(extra_proc_dir, "train_val_test_files.pt"),
        )

        print("Done!")

