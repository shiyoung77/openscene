import os
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch
import open3d as o3d


def process_one_scene(scene_pcd_path):
    scene_name = scene_pcd_path.parent.name
    print(f"{scene_name = }")
    pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    coords = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255) / 127.5 - 1
    labels = 255 * np.ones((coords.shape[0], ), dtype=np.int32)
    save_path = out_dir / f"{scene_name}.pth"
    torch.save((coords, colors, labels), save_path)


if __name__ == "__main__":
    # scene_list = [f'{i:04d}' for i in range(48, 60)]
    # in_path = Path("~/dataset/ycb_video/").expanduser()
    # out_dir = Path("~/dataset/ycb_video_processed/ycb_video_3d").expanduser()

    # scene_list = [f'recording{i}' for i in [4, 8, 9, 10, 11, 12, 13, 14]]
    scene_list = [f'recording{i}' for i in [15, 16, 17, 18]]
    in_path = Path("~/dataset/custom_tabletop/").expanduser()
    out_dir = Path("~/dataset/custom_tabletop_processed/custom_tabletop_3d").expanduser()
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for scene in scene_list:
        files.append(in_path / scene / 'scan-0.005.pcd')

    process_one_scene(files[0])

    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_one_scene, files)
    p.close()
    p.join()
