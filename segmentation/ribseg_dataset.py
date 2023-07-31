# *_*coding:utf-8 *_*

# adapted from https://github.com/M3DV/RibSeg/blob/main/data_utils/dataloader.py


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def pc_normalize(pc, centroid=None, m=None):
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if m is None:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m

    return pc, centroid, m


class RibSegDataset(Dataset):
    def __init__(
        self, root, npoints=30000, split="train", eval=False, binary_root=None
    ):
        # in case of second stage, root should be normal compressed (required for gt), binary_root should be binary compressed
        ignore = [452, 485, 490]

        splits = {
            "train": range(1, 421),
            "val": range(421, 501),
            "test": range(501, 661),
        }
        splits["trainval"] = list(splits["train"]) + list(splits["val"])
        splits["all"] = list(splits["trainval"]) + list(splits["test"])
        assert split in splits
        for key in splits:
            splits[key] = [x for x in splits[key] if x not in ignore]

        self.eval = eval
        self.npoints = npoints
        self.root = root
        # nclasses: 24 ribs + BG
        self.seg_num_all = 25  # first class is degenerate if second_stage is true
        self.second_stage = bool(binary_root)

        ids = splits[split]

        if not self.second_stage:
            self.datapath = [
                os.path.join(self.root, f"RibFrac{idx}.npz") for idx in ids
            ]
        else:
            self.datapath = [
                (
                    os.path.join(self.root, f"RibFrac{idx}.npz"),
                    os.path.join(binary_root, f"RibFrac{idx}.npz"),
                )
                for idx in ids
            ]

    def __getitem__(self, index):
        fn = self.datapath[index]
        if not self.second_stage:
            data = np.load(fn)
            pc = data["seg"].astype(np.float32)
            pc[:, :3], centroid, m = pc_normalize(pc[:, :3])

            ct, label = pc[:, :3], pc[:, 3]
        else:
            data, data_binary = np.load(fn[0]), np.load(fn[1])
            assert np.array_equal(data["input"], data_binary["input"])
            pc = data["input"].astype(
                np.float32
            )  # no need to normalize, since already done in first stage
            label = data["gt"]
            is_valid = (
                data_binary["pred"] > 0
            )  # using the predictions of prev stage to filter background
            # alternative option is
            # is_valid = data["gt"] > 0 # which would use gt to filter background

            ct = pc[is_valid]
            label = label[is_valid]

            fn = fn[0]

        if not self.eval:
            choice = np.random.choice(ct.shape[0], self.npoints, replace=False)
            ct, label = ct[choice], label[choice]
            label = label.astype(int)  # 25 classes, BG + 24 ribs

            return ct, label
        else:
            idx = np.random.permutation(ct.shape[0])
            ct = ct[idx]
            label = label[idx]

            split_idx = np.arange(0, ct.shape[0], self.npoints)[1:]
            ct_list = np.split(ct, split_idx)
            label_list = np.split(label, split_idx)

            n_last = ct_list[-1].shape[0]
            if n_last < self.npoints:
                n_missing = self.npoints - n_last
                fill_idx = np.random.choice(
                    # highly improbable for duplicate points, unless in degenerate cases where small number of points
                    ct.shape[0],
                    n_missing,
                    replace=True
                    # ct.shape[0] - n_last, n_missing, replace=False
                )
                ct_list[-1] = np.concatenate([ct_list[-1], ct[fill_idx]], axis=0)
                label_list[-1] = np.concatenate(
                    [label_list[-1], label[fill_idx]], axis=0
                )

            return (
                fn,
                ct_list,
                label_list,
            )  #  [(ct_list[i], label_list[i]) for i in range(len(ct_list))]

    def __len__(self):
        return len(self.datapath)


def get_centroids_m():
    paths = RibSegDataset(
        root="ribseg_benchmark", split="all", npoints=2048, eval=True
    ).datapath
    results = {}
    for path in tqdm(paths):
        idx = int(path.split("/")[-1].split(".")[0].replace("RibFrac", ""))
        pc = np.load(path)["seg"].astype(
            np.float32
        )  # type casting is necessary to ensure consistency
        _, centroid, m = pc_normalize(pc[:, :3])
        results[idx] = (centroid, m)
    np.savez("centroids_m.npz", centroids_m=results)


if __name__ == "__main__":
    # dataset_a = RibSegDataset(
    #     root="/data/adhinart/ribseg/outputs/dgcnn",
    #     npoints=2048,
    #     eval=False,
    #     binary_root="/data/adhinart/ribseg/outputs/dgcnn_binary",
    # )
    # dataset_b = RibSegDataset(root="ribseg_benchmark", npoints=2048, eval=False)

    # dataset_b = RibSegDataset(root="../ribseg_benchmark", npoints=2048, eval=True)
    # a = dataset[0]

    get_centroids_m()
