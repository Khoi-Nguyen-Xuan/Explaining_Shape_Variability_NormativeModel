import os
import os.path as osp
from glob import glob

import torch
from sklearn.model_selection import train_test_split
import openmesh as om

from torch_geometric.data import Data
from utils.read import read_mesh


class MeshData(object):
    def __init__(self,
                 root,
                 template_fp,
                 split='interpolation',   # kept for compatibility, but unused now
                 test_exp='bareteeth',    # unused for torus
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.template_fp = template_fp
        self.split = split
        self.test_exp = test_exp
        self.transform = transform
        self.pre_transform = pre_transform

        # Datasets will just be Python lists of Data objects
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None

        self.num_train_graph = 0
        self.num_val_graph = 0
        self.num_test_graph = 0

        self.load()

    # -------------------------------------------------------------
    # Main loading logic: build Data lists directly from raw torus
    # -------------------------------------------------------------
    def load(self):
        # ---- 1. Template mesh info ----
        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = int(self.template_face.max()) + 1
        print(f"[MeshData] num_nodes = {self.num_nodes} (from template_face)")

        # ---- 2. Build train/val/test lists from raw & labels ----
        train_list, val_list, test_list = self._build_splits()

        self.train_dataset = train_list
        self.val_dataset   = val_list
        self.test_dataset  = test_list

        self.num_train_graph = len(self.train_dataset)
        self.num_val_graph   = len(self.val_dataset)
        self.num_test_graph  = len(self.test_dataset)

        print(f"[MeshData] Direct build -> train: {self.num_train_graph}, "
              f"val: {self.num_val_graph}, test: {self.num_test_graph}")

        if self.num_train_graph == 0:
            raise RuntimeError("Train dataset is empty in MeshData.load().")

        # ---- 3. Compute per-vertex mean/std from TRAIN ----
        self._compute_mean_std()

        print("[MeshData] mean shape:", self.mean.shape)
        print("[MeshData] std  shape:", self.std.shape)

        # ---- 4. Normalize all splits in-place ----
        print('Normalizing (train/val/test)...')
        self._normalize_list(self.train_dataset, name="train")
        self._normalize_list(self.val_dataset,   name="val")
        self._normalize_list(self.test_dataset,  name="test")
        print('Done!')

    # -------------------------------------------------------------
    # Build splits from raw/torus/*.ply and labels.pt
    # -------------------------------------------------------------
    def _build_splits(self):
        torus_dir = osp.join(self.root, "raw", "torus")
        label_fp = osp.join(torus_dir, "labels.pt")
        print(f"[MeshData] Loading labels from: {label_fp}")
        labels_obj = torch.load(label_fp, map_location="cpu")

        if not isinstance(labels_obj, dict) or "ids" not in labels_obj or "age" not in labels_obj:
            raise RuntimeError(
                "labels.pt must be a dict with keys 'ids' and 'age', "
                f"but got keys: {list(labels_obj.keys())}"
            )

        ids_raw = labels_obj["ids"]
        age_raw = labels_obj["age"]

        import numpy as np
        def to_list(x):
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

        label_map = {sid: float(age) for sid, age in zip(ids_list, age_list)}
        print(f"[MeshData] #subjects in labels: {len(label_map)}")

        # Collect .ply files
        fps = sorted(glob(osp.join(torus_dir, "*.ply")))
        if len(fps) == 0:
            raise RuntimeError(f"No .ply found in {torus_dir}")

        subjects_all = [osp.splitext(osp.basename(fp))[0] for fp in fps]
        print(f"[MeshData] #.ply files: {len(subjects_all)}")

        # Keep only subjects that have labels
        subjects = [s for s in subjects_all if s in label_map]
        print(f"[MeshData] #subjects with labels & meshes: {len(subjects)}")

        if len(subjects) == 0:
            raise RuntimeError(
                "No overlap between .ply basenames and labels['ids']."
            )

        # Train/val/test split on SUBJECT IDs (same as in CoMA.process)
        X_train, X_temp = train_test_split(subjects, test_size=0.2, random_state=28)
        if len(X_temp) >= 2:
            X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=28)
        else:
            X_val, X_test = [], X_temp

        print(
            f"[MeshData] Split sizes -> train: {len(X_train)}, "
            f"val: {len(X_val)}, test: {len(X_test)}"
        )

        # Build PyG Data lists directly
        def build_list(id_list, split_name):
            data_list = []
            for sid in id_list:
                fp = osp.join(torus_dir, f"{sid}.ply")
                if not osp.exists(fp):
                    raise RuntimeError(f"Missing mesh file for subject {sid}: {fp}")
                data = read_mesh(fp)  # geometry only

                # Attach scalar age
                age_value = label_map[sid]
                data.y = torch.tensor([age_value], dtype=torch.float32)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            print(f"[MeshData] {split_name} list size: {len(data_list)}")
            return data_list

        train_list = build_list(X_train, "train")
        val_list   = build_list(X_val,   "val")
        test_list  = build_list(X_test,  "test")

        return train_list, val_list, test_list

    # -------------------------------------------------------------
    # Mean / std over TRAIN graphs (per vertex)
    # -------------------------------------------------------------
    def _compute_mean_std(self):
        V = self.num_nodes
        running_sum = torch.zeros((V, 3), dtype=torch.float32)
        running_sq_sum = torch.zeros((V, 3), dtype=torch.float32)

        for i, data in enumerate(self.train_dataset):
            x_i = data.x
            if x_i.shape[0] != V or x_i.shape[1] != 3:
                raise RuntimeError(
                    f"[MeshData] Train graph {i} has shape {x_i.shape}, expected [V,3] with V={V}."
                )
            running_sum += x_i
            running_sq_sum += x_i * x_i

        mean = running_sum / float(self.num_train_graph)
        var = running_sq_sum / float(self.num_train_graph) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-8))

        self.mean = mean   # [V,3]
        self.std = std     # [V,3]

    # -------------------------------------------------------------
    # Normalize lists in-place
    # -------------------------------------------------------------
    def _normalize_list(self, data_list, name=""):
        V = self.num_nodes
        for i, data in enumerate(data_list):
            x_i = data.x
            if x_i.shape[0] != V or x_i.shape[1] != 3:
                raise RuntimeError(
                    f"[MeshData] {name} graph {i} has shape {x_i.shape}, expected [V,3] with V={V}."
                )
            data.x = (x_i - self.mean) / self.std

    def save_mesh(self, fp, x):
        """
        x: tensor of shape [num_nodes, 3] in normalized space.
        This unnormalizes it and writes a mesh using the template faces.
        """
        x = x * self.std + self.mean  # [num_nodes, 3]
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
