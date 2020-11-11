import torch
from torch.utils.data import Dataset


class sdfDataset(Dataset):
    def __init__(self, sdffile):
        with open(sdffile) as f:
            line = f.readline()
            self.grid_x, self.grid_y, self.grid_z = [int(i) for i in line.split()]

            self.max_xyz = max(self.grid_x, self.grid_y, self.grid_z)

            line = f.readline()
            # self.minxyz = torch.Tensor([float(i) for i in line.split()])

            line = f.readline()
            self.voxel_size = 1.0 / self.max_xyz
            self.orig_voxel = float(line)

            self.minxyz = torch.Tensor([-self.grid_x / 2.0 * self.voxel_size, -self.grid_y / 2.0 * self.voxel_size, -self.grid_z / 2.0 * self.voxel_size])

            max_xyz = max(self.grid_x, self.grid_y, self.grid_z)

            self.idxs = torch.Tensor(
                [[k, j, i, float(f.readline()) * self.voxel_size / self.orig_voxel] for i in range(self.grid_z) for j in range(self.grid_y) for k in
                 range(self.grid_x)])

    def __getitem__(self, idx):
        return self.idxs[idx, :-1] * self.voxel_size + self.minxyz, torch.unsqueeze(self.idxs[idx, -1], 0)

    def __len__(self):
        return self.grid_x * self.grid_y * self.grid_z
