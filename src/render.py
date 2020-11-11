import datetime as dt

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


class Renderer:
    def __init__(self, sdfModel, cameraPos, cameraTarget, screenWidth, screenHeight, rayMarchingTolerence, debug=False):
        self.device = torch.device("cpu" if not torch.cuda.is_available() or debug else "cuda")
        self.sdf = sdfModel.to(self.device) if not debug else lambda x: torch.norm(x) - 3.
        # self.sdf = lambda x: torch.norm(x) - 0.5
        self.camerapos = cameraPos.to(self.device)
        self.cameratarget = cameraTarget.to(self.device)
        self.screen = (screenWidth, screenHeight, 3)
        self.tolerence = rayMarchingTolerence
        self.outImage = torch.zeros(self.screen).to(self.device)
        self.mindist = 1

    def getResult(self):
        return self.outImage

    def rayCast(self, pos, dir):
        dist = 0.
        pre_res = 10
        for i in range(32):
            res = self.sdf((pos + dist * dir).to(self.device))
            # print(res)
            if res < self.tolerence * dist:
                return dist
            if res > pre_res:
                return -1.
            pre_res = res
            dist += res

        return -1.

    def normalizedScreenCoords(self, screenCoord):
        res = 2. * (screenCoord / torch.Tensor([self.screen[0], self.screen[1]]).to(self.device) - 0.5)
        res[1] *= -1.
        res[0] *= self.screen[0] / self.screen[1]
        return res

    def getCameraDir(self, uv):
        camForward = self.cameratarget - self.camerapos
        camForward /= torch.norm(camForward)

        camRight = torch.cross(torch.Tensor([0., 1., 0.]).to(self.device), camForward)
        camRight /= torch.norm(camRight)

        camUp = torch.cross(camForward, camRight).to(self.device)
        camUp /= torch.norm(camUp)

        fPersp = 2.
        vDir = uv[0] * camRight + uv[1] * camUp + fPersp * camForward
        vDir /= torch.norm(vDir)
        return vDir

    def showImage(self):
        res = self.outImage.cpu().numpy()
        cv2.imshow('ss', res)
        cv2.waitKey(0)

    def save(self, name):
        res = self.outImage.cpu().numpy()
        cv2.imwrite(name, res)

    def render(self):

        def renderOne(coord):
            uv = self.normalizedScreenCoords(coord)
            rayDir = self.getCameraDir(uv)

            t = self.rayCast(self.camerapos, rayDir)
            if t >= 0:
                if t < self.mindist:
                    self.mindist = t
                # print(t)
                self.outImage[coord[0].int().item()][coord[1].int().item()] = torch.Tensor(
                    [850 - 410 * t, 850 - 410* t, 850 - 410 * t]).to(self.device)

        coords = torch.Tensor([(i, j) for i in range(self.screen[0]) for j in range(self.screen[1])]).to(self.device)
        for coord in coords:
            # print(coord)
            renderOne(coord)
            # print(self.outImage[coord[0].int().item()][coord[1].int().item()])


if __name__ == "__main__":
    campos = torch.Tensor([0, 1, 2])
    at = torch.Tensor([0, 0, 0])
    width = 1280
    height = 1280
    tol = 0.001
    renderer = Renderer(None, campos, at, width, height, tol, True)
    s = dt.datetime.now()
    renderer.render()
    e = dt.datetime.now()
    print(e - s)
    renderer.showImage()
