#!/usr/bin/env python
import re
import numpy as np
from prody import *
import logging

def getChainResnameResnum(pdb,resIndex):
	# Get a string for chain+resid+resnum when supplied the residue index.
	selection = pdb.select('resindex %i' % resIndex)
	chain = selection.getChids()[0]
	resName = selection.getResnames()[0]
	resNum = selection.getResnums()[0]
	string = chain+resName+str(resNum)
	return string

def getResindex(pdb,chainResnameResnum):
	# Get the residue index of a chain resname resnum string.
	matches = re.search('(\D+)(\D{3})(\d+)',chainResnameResnum)
	if matches:
		chain = matches.groups()[0]
		resName = matches.groups()[1]
		resNum = int(matches.groups()[2])
		selection = pdb.select('chain %s and resnum %i' % (chain,resNum))
		resIndex = selection.getResindices()[0]
		return resIndex

def parseEnergiesSingleCore(filePaths,pdb,logFile):

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)
	# Start a dictionary for storing residue-pair energy values

	energiesDict = dict()
	for filePath in filePaths:
		logger.info('Parsing: '+filePath)
		# Get the interaction residues
		matches = re.search('(\d+)_(\d+)_energies.log',filePath)
		if not matches:
			continue 

		# Get residue indices
		res1 = int(matches.groups()[0])
		res2 = int(matches.groups()[1])

		system = parsePDB(pdb)
		# Get chain-resname-resnum strings
		res1_string = getChainResnameResnum(system,res1)
		res2_string = getChainResnameResnum(system,res2)

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
		energiesDict[res1_string+'-'+res2_string] = energyOutput
		# Also store it as res2,res2 (it is the same thing after all)
		energiesDict[res2_string+'-'+res1_string] = energyOutput

	return energiesDict