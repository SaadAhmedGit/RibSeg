import os
import numpy as np
from tqdm import tqdm


def load_file(idx, binary_dir, binary_second_stage_dir, centroids_m):
    # idx is in range(501,661)
    # binary_dir/binary_second_stage_dir are the paths to the extracted zip files
    # centroids_m is path to the npz file

    idx = int(idx)
    centroids, m = np.load(centroids_m, allow_pickle=True)["centroids_m"].item()[idx]

    name = f"RibFrac{idx}.npz"
    binary = np.load(os.path.join(binary_dir, name))
    second_stage = np.load(os.path.join(binary_second_stage_dir, name))

    assert np.sum(binary["pred"] > 0) == second_stage["pred"].shape[0]
    binary_is_valid = (
        binary["pred"] == 0
    )  # NOTE: since filtering is done based on pred and not seg

    gt = np.concatenate([binary["gt"][binary_is_valid], second_stage["gt"]])
    pred = np.concatenate([binary["pred"][binary_is_valid], second_stage["pred"]])
    input = np.concatenate([binary["input"][binary_is_valid], second_stage["input"]])
    unnormalized_input = input * m + centroids
    unnormalized_input = np.rint(unnormalized_input).astype(int)

    assert np.all(unnormalized_input >= 0)
    assert np.all(unnormalized_input < 2**16)
    unnormalized_input = unnormalized_input.astype(np.uint16)

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    return unnormalized_input, gt, pred


if __name__ == "__main__":
    binary_dir = "/data/adhinart/ribseg/outputs/dgcnn_binary"
    binary_second_stage_dir = "/data/adhinart/ribseg/outputs/dgcnn_binary_second_stage"
    centroids_m = "/data/adhinart/ribseg/centroids_m.npz"

    output_dir = "/data/adhinart/ribseg/outputs/dgcnn_two_stage_preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    ids = list(range(1, 661))
    ignore = [452, 485, 490]
    ids = [i for i in ids if i not in ignore]

    # unnorm_input, gt, pred = load_file(598, binary_dir, binary_second_stage_dir, centroids_m)
    for idx in tqdm(ids):
        # for idx in tqdm(range(501, 661)):
        try:
            unnormalized_input, gt, pred = load_file(
                idx, binary_dir, binary_second_stage_dir, centroids_m
            )
            np.savez(
                os.path.join(output_dir, f"RibFrac{idx}.npz"),
                input=unnormalized_input,
                gt=gt,
                pred=pred,
            )
        except:
            print(f"Error in {idx}")
            __import__("pdb").set_trace()
