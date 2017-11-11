# gRINN: Get Residue Interaction eNergies and Networks

version: 0.0.1dev

date: 2017/10/25

Marmara University

Department of Bioengineering

Computational Biology and Bioinformatics Group

developer: Onur Serçinoğlu

contact: onursercin@gmail.com


## INTRODUCTION

This package contains three main programs:

* **getResIntEn.py** is a Python command-line application for calculating pairwise amino-acid interaction
energies from NAMD-generated Molecular Dynamics simulation trajectories.

* **getResIntCorr.py** is a Python command-line application for calculating equal-time linear correlation
between amino-acid interaction energy time profiles from the output of getResIntEn.py

* **getResIntEnGUI.py** is a PyQt5-based Graphical User Interface (GUI) for getResIntEn.py and getResIntCorr.py

All of the functionality contained in this package can be accessed via command-line interface of 
getResIntEn.py. For more help regarding the use of this command, open a terminal and type:
python3 getResIntEn.py --help

## DEPENDENCIES

All of the packages require Visual Molecular Dynamics (VMD) program to be accessed readily from terminal.
To see whether this requirement is already met or not, open a terminal and type:
vmd

If you see the VMD Main Window starting, then you're good to go.
To download and install VMD, please visit http://www.ks.uiuc.edu/Research/vmd/

Residue interaction energy calculations require a NAMD binary, whose path can be specified via argument 
--namd2exe /path/to/namd2 to getResIntEn.py or selected via getResIntEnGUI.py window.

All of the packages require python 3 or above. 

Other dependencies:

getResIntEn.py requires ProDy 1.8.2, pyprind, numpy, psutil, signal, datetime, argparse, itertools.

getResIntCorr.py additionally requires SciPy.

getResIntCorrGUI.py additionally requires PyQt5.

After installing all dependencies, simply clone this repository to start using it.

## USAGE

This package is intended for VMD/NAMD users who would like to calculate pairwise interaction energies 
between individual amino acids in the protein structure after completing a Molecular Dynamics simulation.

For such calculation PDB (coordinates), PSF (topology) and DCD (trajectory) files are required.
If explicit solvent is used in the simulation (which is usually the case), it is highly recommended to
delete all waters in these files prior to the use of any tool in this package. Inclusion of water will
almost certainly cause rapid memory depletion especially for large file sizes, causing the system to 
freeze. 

After preparing the PDB/PSF/DCD files, getResIntEn.py can be accessed either via the terminal or the GUI.
All of the options that can be specified via the terminal can also be specified via the GUI.

Via terminal: **python3 getresinten/getResIntEn.py <options>**

type `python3 getresinten/getResIntEn.py --help` to see the full list of options.

Via GUI: **python3 getresinten/getResIntEnGUI.py**

The package includes a program to calculate residue interaction energy correlations as well: **getResIntCorr**. 
This can be used separately or used when calling **getResIntEn.py**

Interaction energies and Interaction Energy Matrices (IEM) can be viewed with: 

`python3 getresinten/viewResults.py``

IEM and Residue Correlation (RC) matrices can be used to construct "Protein Energy Networks" by calling:

`python3 getresinten/getProEnNet.py``

type `python3 getresinten/getProEnNet.py --help` to see the full list of options.

## TUTORIAL

A tutorial is available at https://bitbucket.org/onursercinoglu/getresinten/wiki/Home