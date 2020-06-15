# dolfin_cm
A set of FEniCS-based python tools for Computational Mechanics.
### Requirements
First you need to install [myPythonLibrary](https://gitlab.inria.fr/mgenet/myPythonLibrary) and [myVTKPythonLibrary](https://gitlab.inria.fr/mgenet/myVTKPythonLibrary). You also need a working installation of FEniCS (including DOLFIN python interface).
### Installation
Get the code:
```
git clone https://gitlab.inria.fr/mgenet/dolfin_cm.git
```
To be able to load the library within python, the simplest is to add the folder containing `dolfin_cm` to the `PYTHONPATH` environment variable:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder
```
(To make this permanent, add the line to `~/.bashrc`.)
Then you should be able to load the library within python:
```
import dolfin_cm as dcm
```
