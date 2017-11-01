#!/usr/bin/env python
import re
import numpy as np

def parseEnergiesSingleCore(filePaths):

	# Start a dictionary for storing residue-pair energy values
	energiesDict = dict()
	for filePath in filePaths:
		# Get the interaction residues
		matches = re.search('(\d+)_(\d+)_energies.log',filePath)
		if not matches:
			continue 

		# Important!!! Converting from Tcl 0-based indexing to 1 based indexing (more logical.)
		res1 = int(matches.groups()[0])+1 
		res2 = int(matches.groups()[1])+1

		# Read in the first line (header) output file and count number of total lines.
		f = open(filePath,'r')
		lines = f.readlines()

		# Ignore lines not starting with ENERGY:
		lines = [line for line in lines if line.startswith('ENERGY:')]
		f.close()

		lines = [line.strip('\n').split() for line in lines if line.startswith('ENERGY:')]
		lines = [[float(integer) for integer in line[1:]] for line in lines]

		headers = ['Frame','Elec','VdW','Total']
		headerColumns = [0,5,6,10] # Order in which the headers appear in NAMD2 log
		# Frame: 0, Elec: 5, VdW: 6, Total: 10
		numTitles = len(headers)
		# Assign each column into appropriate key's value in energyOutput dict.
		energyOutput = dict()
		for i in range(0,numTitles):
			energyOutput[headers[i]] = [line[headerColumns[i]] for line in lines]

		# Puts this energyOutput dict into energies dict with keys as residue ids
		energiesDict[(res1,res2)] = energyOutput
		# Also store it as res2,res2 (it is the same thing after all)
		energiesDict[(res2,res1)] = energyOutput

	return energiesDict