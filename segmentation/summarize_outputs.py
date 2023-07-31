import os
import sys
import numpy as np
import nibabel as nib
import h5py

from tqdm import tqdm

import argparse

import multiprocessing as mp


def task(input_dir, file, output_dir):
    try:
        output_name = os.path.join(output_dir, file)
        if os.path.exists(output_name) and not OVERWRITE:
            return
        data = np.load(os.path.join(input_dir, file))

        input = data["data"]
        pred = data["pred"]
        pred = np.argmax(pred, axis=1)  # get actual idx instead of weights
        gt = data["seg"].astype(pred.dtype)

        input = input.reshape(-1, 3)

        unique_idx = np.unique(input, axis=0, return_index=True)[1]
        input, pred, gt = input[unique_idx], pred[unique_idx], gt[unique_idx]

        np.savez_compressed(output_name, input=input, pred=pred, gt=gt)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Args: {input_dir}, {file}, {output_dir}")
        raise e


def compress_inputs(input_dir, output_dir):
    tasks = [(input_dir, file, output_dir) for file in os.listdir(input_dir)]
    with mp.Pool(16) as pool:
        tasks = list(tqdm(pool.starmap(task, tasks), total=len(tasks)))


def task_nnunet(pred_dir, label_dir, file, output_dir):
    try:
        output_name = (
            os.path.join(output_dir, file).replace(".nii.gz", ".npz").replace("_", "")
        )
        if os.path.exists(output_name) and not OVERWRITE:
            return
        pred = nib.load(os.path.join(pred_dir, file)).get_fdata().astype(np.uint16)
        gt = nib.load(os.path.join(label_dir, file)).get_fdata().astype(np.uint16)

        input = np.meshgrid(
            np.arange(pred.shape[0]),
            np.arange(pred.shape[1]),
            np.arange(pred.shape[2]),
            indexing="ij",
        )
        input = np.stack(input, axis=-1).reshape(-1, 3).astype(np.uint16)
        pred, gt = pred.reshape(-1), gt.reshape(-1)

        np.savez_compressed(output_name, input=input, pred=pred, gt=gt)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Args: {pred_dir} {label_dir} {file} {output_dir}")
        raise e


def compress_inputs_nnunet(pred_dir, label_dir, output_dir):
    # RibFrac_514.nii.gz
    preds = [x for x in sorted(os.listdir(pred_dir)) if x.endswith(".nii.gz")]
    labels = [x for x in sorted(os.listdir(label_dir)) if x.endswith(".nii.gz")]

    assert preds == labels

    tasks = [(pred_dir, label_dir, file, output_dir) for file in preds]
    with mp.Pool(16) as pool:
        tasks = list(tqdm(pool.starmap(task_nnunet, tasks), total=len(tasks)))


def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    dirs = {}
    dirs["pointcnn"] = "/data/adhinart/ribseg/pointcnn_pytorch/outputs/normal"
    dirs["pointcnn_binary"] = "/data/adhinart/ribseg/pointcnn_pytorch/outputs/binary"
    dirs[
        "pointcnn_binary_second_stage"
    ] = "/data/adhinart/ribseg/pointcnn_pytorch/outputs/second_stage"
    dirs[
        "dgcnn"
    ] = "/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32/inference"
    dirs[
        "dgcnn_binary"
    ] = "/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary/inference"
    dirs[
        "dgcnn_binary_second_stage"
    ] = "/data/adhinart/ribseg/dgcnn.pytorch/outputs/ribseg_2048_40_32_binary_second_stage/inference"

    dirs[
        "pointnet1"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1/outputs"
    dirs[
        "pointnet1_binary"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary/outputs"
    dirs[
        "pointnet1_binary_second_stage"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_binary_second_stage/outputs"
    dirs[
        "pointnet2"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2/outputs"
    dirs[
        "pointnet2_binary"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary/outputs"
    dirs[
        "pointnet2_binary_second_stage"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_binary_second_stage/outputs"

    dirs[
        "pointnet1_2048"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048/outputs"
    dirs[
        "pointnet1_2048_binary"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary/outputs"
    dirs[
        "pointnet1_2048_binary_second_stage"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet1_2048_binary_second_stage/outputs"
    dirs[
        "pointnet2_2048"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048/outputs"
    dirs[
        "pointnet2_2048_binary"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary/outputs"
    dirs[
        "pointnet2_2048_binary_second_stage"
    ] = "/data/adhinart/ribseg/Pointnet_Pointnet2_pytorch/log/ribseg/pointnet2_2048_binary_second_stage/outputs"

    dirs["nnunet_3d_fullres"] = (
        "/scratch/adhinart/nnUNet_results/fullres_inference",
        "/scratch/adhinart/nnUNet_raw/Dataset011_RibSeg/labelsTs",
    )

    print(dirs)

    for name, input_dir in dirs.items():
        output_dir = os.path.join("outputs", name)
        os.makedirs(output_dir, exist_ok=True)

        if not "nnunet" in name:
            compress_inputs(input_dir, output_dir)
        else:
            compress_inputs_nnunet(input_dir[0], input_dir[1], output_dir)


OVERWRITE = True
if __name__ == "__main__":
    print(f"Overwrite: {OVERWRITE}")
    main()
