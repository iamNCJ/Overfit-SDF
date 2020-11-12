# Overfit-SDF
Using neural network to overfit the SDF shape representation

> Our final project for course ***Intelligent Accusation of Visual Information***

## Prerequisite Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

```bash
usage: NeuralImplicit.py [-h] [--input INPUT_SDF] [--verbose]
                         [--render RENDER_MODEL] [--headless]

Overfit an implicit neural network to represent 3D shape, type --help to see
available arguments

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT_SDF     The SDF file to overfit
  --verbose, -v         Train in verbose mode
  --render RENDER_MODEL
                        The pth model file to load and render
  --headless            Render in headless mode
```

### Examples

```bash
# Overfit
python3 network/NeuralImplicit.py --input input.sdf
# Render
python3 network/NeuralImplicit.py --render input.pth
```

## Data Preprocessing - Generating SDF from Mesh
If you have a mesh file xxx.obj, you need to generate SDF from the mesh file to run our SDFDiff code.

First, you need to git clone the following tools.

```bash
# a tool to generate watertight meshes from arbitrary meshes
git clone https://github.com/hjwdzh/ManifoldPlus.git

# A tool to generate SDF from watertight meshes
git clone https://github.com/christopherbatty/SDFGen.git
```

Then you can run the following to get SDF from your mesh file xxx.obj.

```bash
# Generate watertight meshes from arbitrary meshes
./ManifoldPlus/build/manifold --input ./obj_files/xxx.obj --output ./watertight_meshes_and_sdfs/xxx.obj

# Generate SDF from watertight meshes
./SDFGen/build/bin/SDFGen ./watertight_meshes_and_sdfs/xxx.obj 0.002 0 
```

## Acknowledgements and References

- Davies, Thomas, Derek Nowrouzezahrai, and Alec Jacobson. “Overfit Neural Networks as a Compact Shape Representation.” *ArXiv:2009.09808 [Cs]*, October 12, 2020. http://arxiv.org/abs/2009.09808.

- Park, Jeong Joon, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. “DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.” *ArXiv:1901.05103 [Cs]*, January 15, 2019. http://arxiv.org/abs/1901.05103.

- Jiang, Yue, Dantong Ji, Zhizhong Han, and Matthias Zwicker. “SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization.” *ArXiv:1912.07109 [Cs]*, December 15, 2019. http://arxiv.org/abs/1912.07109.

- Huang, Jingwei, Hao Su, and Leonidas Guibas. “Robust Watertight Manifold Surface Generation Method for ShapeNet Models.” *ArXiv:1802.01698 [Cs]*, February 5, 2018. http://arxiv.org/abs/1802.01698.

- Huang, Jingwei, Yichao Zhou, and Leonidas Guibas. “ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups.” *ArXiv:2005.11621 [Cs]*, May 23, 2020. http://arxiv.org/abs/2005.11621.