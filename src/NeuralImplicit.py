import os
import time
from torch import nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sdfDataset import sdfDataset

class NeuralImplicit:
    def __init__(self, N=16, H=64):
        self.model = self.OverfitSDF(N, H)
        self.epochs = 1000
        self.lr = 1e-4
        self.batch_size = 64
        self.log_iterations = 1000

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        print('loading model...')
        self.model.load_state_dict(torch.load(name))

    # Supported mesh file formats are .obj and .stl
    # Sampler selects oversample_ratio * num_sample points around the mesh, keeping only num_sample most
    # important points as determined by the importance metric
    def encode(self, mesh_file, num_samples=1000000, verbose=True):
        # dataset = self.MeshDataset(mesh_file, num_samples, oversample_ratio)
        dataset = sdfDataset(mesh_file)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        loss_func = nn.L1Loss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for e in range(self.epochs):
            epoch_loss = 0
            eval_loss = 0
            self.model.train()
            count = 0
            for batch_idx, (x_train, y_train) in enumerate(dataloader):
                x_train, y_train = x_train.to(device), y_train.to(device)
                count += self.batch_size
                optimizer.zero_grad()

                y_pred = self.model(x_train)

                loss = loss_func(y_pred, y_train)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if (verbose and count % 10000 == 0):
                    msg = '{}\tEpoch: {}:\t[{}/{}]\tepoch_loss: {:.6f}\tloss: {:.6f}'.format(
                        time.ctime(),
                        e + 1,
                        count,
                        len(dataset),
                        epoch_loss / (batch_idx + 1),
                        loss)
                    print(msg)

            # with self.model.eval():
            #     pass

            print('Saving model...')
            model_file = "./_" + os.path.splitext(os.path.basename(mesh_file))[0] + ".pth"
            self.save(model_file)


    # The actual network here is just a simple MLP
    class OverfitSDF(nn.Module):
        def __init__(self, N, H):
            super().__init__()
            assert (N > 0)
            assert (H > 0)

            net = [nn.Linear(3, H), nn.ReLU(True)]

            for i in range(0, N):
                net += [nn.Linear(H, H), nn.ReLU(True)]

            net += [nn.Linear(H, 1)]
            self.model = nn.Sequential(*net)

        def forward(self, x):
            x = self.model(x)
            output = torch.tanh(x)
            return output

if __name__ == '__main__':
    print(torch.cuda.is_available())

    bunny = NeuralImplicit()
    print(torch.cuda.device_count())
    bunny.encode('bunny.sdf')
