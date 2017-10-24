#!/usr/bin/env python
import re
import numpy as np

def parseEnergiesSingleCore(filePaths):

	# Start a dictionary for storing residue-pair energy values
	energiesDict = dict()
	for filePath in filePaths:
		# Get the interaction residues
		matches = re.search('(\d+)_(\d+)_energies.dat',filePath)
		if not matches:
			continue 

		# Important!!! Converting from Tcl 0-based indexing to 1 based indexing (more logical.)
		res1 = int(matches.groups()[0])+1 
		res2 = int(matches.groups()[1])+1

		# Read in the first line (header) output file and count number of total lines.
		f = open(filePath,'r')
		lines = f.readlines()
		numLines = len(lines)
		header = lines[0].split()
		numTitles = len(header)

		# Close the file and reopen it with numpy's loadtxt, which is much more practical for numeric data.
		f.close()
		energies = np.loadtxt(filePath,skiprows=1) # Skipping the header row.

		# Assign each column into appropriate key's value in energyOutput dict.
		energyOutput = dict()
		for i in range(0,numTitles):
			energyOutput[header[i]] = energies[:,i]

		# Puts this energyOutput dict into energies dict with keys as residue ids
		energiesDict[(res1,res2)] = energyOutput
		# Also store it as res2,res2 (it is the same thing after all)
		energiesDict[(res2,res1)] = energyOutput

	return energiesDict