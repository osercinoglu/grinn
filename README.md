# gRINN: get Residue Interaction eNergies and Networks

gRINN is a software for residue interaction enery-based analysis of protein MD simulation trajectories.

## Version

1.1.0

## Authors

Onur Serçinoğlu (onursercin@gmail.com)

Pemra Ozbek (pemra.ozbek@marmara.edu.tr)

## License

Please read the license.rst file for usage terms and conditions. 

## Usage

No need to install. Just open a terminal in the folder containing the grinn executable and start grinn by typing ./grinn.

You can also place grinn to a folder included in your executable search path and call grinn from any other directory.

## Dependencies

gRINN depends on either NAMD or GROMACS. Please see http://grinn.readthedocs.io/en/latest/download.html for more details.

## Availability

gRINN is available for Linux x64 and MacOSX operating systems as a standalone executable. 

## Documentation

Documentation for gRINN is located at http://grinn.readthedocs.io

## Tutorial

Best way to learn about the features of gRINN and how to use it is to follow the tutorial at 
http://grinn.readthedocs.io/en/latest/tutorial.html

## History/Change Log

v1.1.0.hf1 (2018/06/21)
^^^^^^^^^^^^^^^^^^^^^^^

This hf (hot-fix) version fixes two bugs which rendered gRINN unusable in some cases and an addition to sample input files:

Bug fixes:

* A major bug in gRINN which leads to a failure in processing TPR files without chain IDs is corrected. gRINN will assign a default chain ID of "P" to residues which have no chain IDs assigned in input TPR.
* IEM annotation is now shown only for smallest proteins (with sizes of at most 20 amino acids).

Additions:

* charm27.ff files (used by GROMACS sample trajectory data) are included in the distribution (considering that this force-field may not be included in GROMACS installation of some users).


v1.1.0 (2018/04/06)
^^^^^^^^^^^^^^^^^^^

This version introduces a major internal code rehaul, leaving major features of gRINN unaffected.
There are additional new features as well as minor bug fixes:

New Features:

* A new calculation setting for non-bonded interaction cutoff for NAMD simulation input is introduced. In the previous version, filtering cutoff distance parameter specified for filtering cutoff distance and non-bonded cutoff for NAMD simulation input.

* gRINN now supports Charmm simulation input as well.

Minor bug fixes:

* A bug which cause multiple parameter files reading to fail for NAMD simulation input is fixed.

* A minor bug which caused incorrect protein structure display upon start of View Results interface in Mac OS version is fixed.

### v1.0.1 (2017/12/27)

Initial release of gRINN.

## Credits

gRINN was coded in Python 2.7.

Several open-source packages, including ProDy, MDTraj, PyQt5, matplotlib, seaborn, pandas, networkx and PyMol are distributed with gRINN. More details can be found in license.rst.

A full list of credits can be found in http://grinn.readthedocs.io/en/latest/credits.html.