#!/usr/bin/env python
import re, os, pexpect, panedr, pandas
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

def parseEnergiesSingleCoreNAMD(filePaths,pdb,logFile):

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

def parseEnergiesGMX(gmxExe,pdb,outputFolder,pairsFiltered):

	system = parsePDB(pdb)
	system_dry = system.select('protein or nucleic')
	system_dry = system_dry.select('not resname SOL')
	gmxExe = 'gmx' # TEMPORARY!
	# Parse the resulting interact.edr file from the output directory

	# WITHOUT PANEDR
	# proc = pexpect.spawnu('%s energy -f %s/interact.edr -o %s/interact.xvg' % (gmxExe,outputFolder,outputFolder))
	# for pair in pairsFiltered:
	# 	res1 = str(pair[0])
	# 	res2 = str(pair[1])
	# 	proc.send('LJ-SR:res%s-res%s' % (res1,res2))
	# 	proc.sendline()
	# 	proc.send('Coul-SR:res%s-res%s' % (res1,res2))
	# 	proc.sendline()
	# proc.sendline()
	# proc.sendline()
	# proc.kill(1)

	# WITH PANEDR
	df = panedr.edr_to_df(outputFolder+'/interact0.edr')
	for i in range(1,len(pairsFiltered)):
		df_pair = panedr.edr_to_df(outputFolder+'/interact'+str(i)+'.edr')
		df = pandas.concat([df,df_pair],axis=1)

	energiesDict = dict()
	for pair in pairsFiltered:
		res1_string = getChainResnameResnum(system_dry,pair[0])
		res2_string = getChainResnameResnum(system_dry,pair[1])
		energyDict = dict()
		energyDict['VdW'] = df['LJ-SR:res%i-res%i' % (pair[0],pair[1])].values
		energyDict['Elec'] = df['Coul-SR:res%i-res%i' % (pair[0],pair[1])].values
		energyDict['Total'] = [energyDict['VdW'][i]+energyDict['Elec'][i] for i in range(0,len(energyDict['VdW']))]

		key1 = res1_string+'-'+res2_string
		key1 = key1.replace(' ','')
		key2 = res2_string+'-'+res1_string
		key2 = key2.replace(' ','')
		energiesDict[key1] = energyDict
		energiesDict[key2] = energyDict

	return energiesDict
	print('Parsing success!')

def makeNDXMDPforGMX(gmxExe='gmx',pdb=None,tpr=None,pairsFiltered=None,sourceSel=None,targetSel=None,outFolder=os.getcwd()):
	
	system = parsePDB(pdb)

	# Modify atom serial numbers to account for possible PDB files with more than 99999 atoms
	system.setSerials(np.arange(1,system.numAtoms()+1))
	
	system_dry = system.select('protein or nucleic')
	system_dry = system_dry.select('not resname SOL')

	if not pairsFiltered and sourceSel and targetSel: # For use outside of getResIntEn

		# Get the source & target selection.
		source = system_dry.select(sourceSel)
		target = system_dry.select(targetSel)

		# For each individual residue included in this selection, get serials, dump them in a dictionary
		# Divide atom serials into chunks of size 15 (this is how index files are made by GMX apparently)
		sourceResIndices = np.unique(source.getResindices())
		sourceResSerials= dict()

		for index in sourceResIndices:
			residue = source.select('resindex %i' % index)
			lenSerials = len(residue.getSerials())
			if lenSerials > 14:
				residueSerials = residue.getSerials()
				sourceResSerials[index] = [residueSerials[i:i+14] for i in range(0,lenSerials,14)]
			else:
				sourceResSerials[index] = np.asarray([residue.getSerials()])


		# Do the same stuff for the target selection.
		# Get the source selection.
		targetResIndices = np.unique(target.getResindices())
		targetResSerials = dict()

		for index in targetResIndices:
			residue = target.select('resindex %i' % index)
			lenSerials = len(residue.getSerials())
			if lenSerials > 14:
				residueSerials = residue.getSerials()
				targetResSerials[index] = [residueSerials[i:i+14] for i in range(0,lenSerials,14)]
			else:
				targetResSerials[index] = np.asarray([residue.getSerials()])

		# Merge the two dicts.
		allSerials = dict()
		allSerials.update(sourceResSerials)
		allSerials.update(targetResSerials)

	else:

		indicesFiltered = np.unique(np.hstack(pairsFiltered))
		allSerials = dict()

		for index in indicesFiltered:
			residue = system_dry.select('resindex %i' % index)
			lenSerials = len(residue.getSerials())
			if lenSerials > 14:
				residueSerials = residue.getSerials()
				allSerials[index] = [residueSerials[i:i+14] for i in range(0,lenSerials,14)]
			else:
				allSerials[index] = np.asarray([residue.getSerials()])

	# Write a standart .ndx file for GMX
	filename = str(outFolder)+'/interact.ndx'

	proc = pexpect.spawnu('%s make_ndx -f %s -o %s' % (gmxExe,tpr,filename))
	proc.send('q')
	proc.sendline()
	proc.sendline()
	proc.wait()
	proc.kill(1)

	# Append our residue groups to this standart file!
	f = open(filename,'a')
	for key in allSerials:
		f.write('[ res%i ]\n' % key)
		if type(allSerials[key][0]).__name__ == 'ndarray':
			for line in allSerials[key][0:]:
				f.write(' '.join(list(map(str,line)))+'\n')
		else:
			f.write(' '.join(list(map(str,allSerials)))+'\n')
	#f.close()

	# Write the .mdp files necessary for GMX

	i = 0
	for pair in pairsFiltered:
		filename = str(outFolder)+'/interact'+str(i)+'.mdp'
		f = open(filename,'w')
		f.write('cutoff-scheme = group\n')
		resString = ''
		resString += 'res'+str(pair[0])+' '
		resString += 'res'+str(pair[1])+' '

		resString += ' SOL'

		f.write('energygrps = '+resString+'\n')

		# Add energygroup exclusions.
		energygrpExclString = 'energygrp-excl ='

		# GOTTA COMMENT OUT THE FOLLOWING DUE TO TOO LONG LINE ERROR IN GROMPP
		# for key in allSerials:
		# 	energygrpExclString += ' res%i res%i' % (key,key)

		energygrpExclString += ' SOL SOL'
		f.write(energygrpExclString)

		f.close()
		i += 1

