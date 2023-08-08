import shutil
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2


def main():
    # in_dir = Path("~/dataset/ycb_video").expanduser()
    # out_dir = Path("~/dataset/ycb_video_processed/ycb_video_2d").expanduser()
    in_dir = Path("~/dataset/custom_tabletop").expanduser()
    out_dir = Path("~/dataset/custom_tabletop_processed/custom_tabletop_2d").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsic = np.eye(4)
    intrinsic[0][0] = 604.5112
    intrinsic[1][1] = 604.8323
    intrinsic[0][2] = 327.6927
    intrinsic[1][2] = 246.4001
    np.savetxt(out_dir / 'intrinsics.txt', intrinsic)

    # scene_list = [f'{i:04d}' for i in range(48, 60)]
    # scene_list = [f'recording{i}' for i in [4, 8, 9, 10, 11, 12, 13, 14]]
    scene_list = [f'recording{i}' for i in [15, 16, 17, 18]]
    for scene in tqdm(scene_list):
        out_scene_dir = out_dir / scene
        out_color_dir = out_scene_dir / "color"
        out_depth_dir = out_scene_dir / "depth"
        out_pose_dir = out_scene_dir / "pose"
        out_color_dir.mkdir(parents=True, exist_ok=True)
        out_depth_dir.mkdir(parents=True, exist_ok=True)
        out_pose_dir.mkdir(parents=True, exist_ok=True)

        in_color_dir = in_dir / scene / "color"
        for color_path in in_color_dir.iterdir():
            # color_im = cv2.imread(str(color_path))
            out_color_path = out_color_dir / f'{int(color_path.name.split("-")[0])}.jpg'
            # cv2.imwrite(str(out_color_path), color_im)
            shutil.copy(color_path, out_color_path)

        in_depth_dir = in_dir / scene / "depth"
        for depth_path in in_depth_dir.iterdir():
            out_depth_path = out_depth_dir / f'{int(depth_path.name.split("-")[0])}.png'
            shutil.copy(depth_path, out_depth_path)

        in_pose_dir = in_dir / scene / "poses"
        for pose_path in in_pose_dir.iterdir():
            out_pose_path = out_pose_dir / f'{int(pose_path.name.split("-")[0])}.txt'
            shutil.copy(pose_path, out_pose_path)


if __name__ == "__main__":
    main()
