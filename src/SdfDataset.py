import torch
from torch.utils.data import Dataset

class SdfDataset(Dataset):
    def __init__(self, sdf_file):
        with open(sdf_file) as f:
            line = f.readline()
            self.grid_x, self.grid_y, self.grid_z = [int(i) for i in line.split()]
            max_xyz = max(self.grid_x, self.grid_y, self.grid_z)

            f.readline()  # ignore origin grid position

            line = f.readline()
            self.voxel_size = 1.0 / max_xyz
            self.orig_voxel = float(line)

            self.min_xyz = torch.Tensor([
                -self.grid_x / 2.0 * self.voxel_size,
                -self.grid_y / 2.0 * self.voxel_size,
                -self.grid_z / 2.0 * self.voxel_size])

            self.indices = torch.Tensor(
                [[k, j, i, float(f.readline()) * self.voxel_size / self.orig_voxel]
                 for i in range(self.grid_z)
                 for j in range(self.grid_y)
                 for k in range(self.grid_x)])

    def __getitem__(self, idx):
        return self.indices[idx, :-1] * self.voxel_size + self.min_xyz, torch.unsqueeze(self.indices[idx, -1], 0)

    def __len__(self):
        return self.grid_x * self.grid_y * self.grid_z
