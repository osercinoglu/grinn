#!/usr/bin/env python

import networkx as nx 
from prody import *
import numpy as np
import pyprind
import argparse

def getKongKarplusNetwork(resIntCorrFile,resCorrFile,pdb,resMeanIntEnFile=False,includeCovalents=True,
	corrCutoff=0,intEnCutoff=0,outName='resNetwork'):

	# Get the number of residues
	sys = parsePDB(pdb)
	numResidues = sys.numResidues()

	# Create the network, and add nodes representing all residues in the protein
	network = nx.Graph()

	for i in range(0,numResidues):
		network.add_node(i+1)

	# Load the resCorrFile and determine the maximum edge weight
	resCorrMat = np.loadtxt(resCorrFile)
	resCorrArray = np.squeeze(resCorrMat)
	maxResCorr = np.max(resCorrArray)

	# Load average residue interaction matrix, if provided.
	if resMeanIntEnFile:
		resIntEnMat = np.loadtxt(resMeanIntEnFile)

	# If covalent bonds are also requested, then
	# Connect covalently bound residues with edge distance of -log(maxResCorr)
	if includeCovalents:
		for i in range(0,numResidues-1):
			res1 = sys.select('resindex %i' % i)
			res2 = sys.select('resindex %i' % (i+1))
		
			# Only connect if the two residues are in the same chain
			if (res1.getChids()[0] == res2.getChids()[0]) and (res1.getSegindices()[0] == res2.getSegindices()[1]):
				network.add_edge(i+1,i+2,{'distance':float(maxResCorr)})

	# Load the resIntCorrFile, get interacting residue pairs and add edges between them
	# Load the resCorrFile and add edge weights to previously added edges by making them equal to
	# absolute correlation values. (This only works if includeCovalents is True)
	f = open(resIntCorrFile,'r')
	lines = f.readlines()

	progbar = pyprind.ProgBar(len(lines))

	for i in range(0,len(lines)):
		line = lines[i].split()
		
		res1 = int(line[0])
		res2 = int(line[1])
		res3 = int(line[2])
		res4 = int(line[3])

		# Check whether edges exist between these residues.
		for [res1_pair,res2_pair] in [[res1,res2],[res3,res4]]:
			if not network.has_edge(res1_pair,res2_pair):

			# Check whether the correlation between the two residues is above corrCutoff.
				if float(resCorrMat[res1_pair-1,res2_pair-1]) >= float(abs(corrCutoff)):

					# Check wheter the user also requested an energy cutoff.
					# If yes, then add an edge only if the mean interaction energy is above the cutoff.
					if resMeanIntEnFile:
						if np.abs(resIntEnMat[res1_pair-1][res2_pair-1]) > intEnCutoff:
							network.add_edge(res1_pair,res2_pair,
								{'distance':float(resCorrMat[res1_pair-1,res2_pair-1])})

		progbar.update()

	# Write the network to several file formats readable by network analysis packages?
	nx.write_gml(network,outName+'KongKarplus'+'.gml')

	return network

def getRibeiroOrtizNetwork(pdb,resMeanIntEnFile=False,includeCovalents=True,intEnCutoff=1,rmsdEn=5,
	outName='resNetwork'):

	# Get the number of residues
	sys = parsePDB(pdb)
	numResidues = sys.numResidues()

	# Create the network, and add nodes representing all residues in the protein
	network = nx.Graph()

	for i in range(0,numResidues):
		network.add_node(i+1)

	# Load average residue interaction matrix.
	resIntEnMat = np.loadtxt(resMeanIntEnFile)

	# Determine RMSD of interaction energies (needed later on)
	resIntEnArray = [i for i in np.abs(np.reshape(resIntEnMat,(1,numResidues**2))[0]) if i > 0]
	rmsdIntEn = np.sqrt(((resIntEnArray - np.mean(resIntEnArray)) ** 2).mean())

	# Construct an matrix to make edge weights later on according to Ribeiro et al. (2014)
	X = 0.5*(1-(resIntEnMat-np.mean(resIntEnArray))/(5*rmsdIntEn))

	for i in range(0,numResidues):
		for j in range(0,numResidues):

			if X[i,j] > 0.99:
				X[i,j] = 0.99

	# If covalent bonds are also requested, then
	# Connect covalently bound residues with edge weight of 0.99
	if includeCovalents:
		for i in range(0,numResidues-1):
			res1 = sys.select('resindex %i' % i)
			res2 = sys.select('resindex %i' % (i+1))
		
			# Only connect if the two residues are in the same chain
			# Weights as is
			if (res1.getChids()[0] == res2.getChids()[0]) and (res1.getSegindices()[0] == res2.getSegindices()[1]):
				network.add_edge(i+1,i+2,{'distance':X[i,i+1]})


			# Weights with -log
			#if (res1.getChids()[0] == res2.getChids()[0]) and (res1.getSegindices()[0] == res2.getSegindices()[1]):
			#	network.add_edge(i+1,i+2,{'distance':-np.log(0.99)})

	progbar = pyprind.ProgBar(385**2)

	for i in range(0,numResidues):
		for j in range(0,numResidues):

			if not includeCovalents:
			# Check again for covalent connection. If we are iterating over a residue pair covalently connected, then skip
				if abs(i-j) == 1:
					continue

		# Check whether edges exist between these residues.
			if not network.has_edge(i+1,j+1):

				# Check whether the mean interaction energy between the two residues is above the cutoff value.
				# If yes, continue.
				if abs(float(resIntEnMat[i,j])) >= float(abs(intEnCutoff)):

					# Add an edge between the two residues. Specify distance according to Ribeiro et al. (2014)
					# Also, consider edge weights lower than 0.01 disconnected
					# (again, Ribeiro et al. 2014)
					if X[i,j] < 0.01:
						continue

					# weights as is
					network.add_edge(i+1,j+1,{'distance':X[i,j]})
					# weights as -log
					#network.add_edge(i+1,j+1,{'distance':-np.log(X[i,j])})

		progbar.update()

	# Write the network to several file formats readable by network analysis packages?
	nx.write_gml(network,outName+'RibeiroOrtiz'+'.gml')

	return network


def convert_arg_line_to_args(arg_line):
	# To override the same method of the ArgumentParser (to read options from a file)
	# Credit and source: hpaulj from StackOverflow
	# http://stackoverflow.com/questions/29111801/using-fromfile-prefix-chars-with-multiple-arguments-nargs#
	for arg in arg_line.split():
		if not arg.strip():
			continue
		yield arg

if __name__ == '__main__':

	# INPUT PARSING
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
		description='Construct Protein Energy Networks from getResIntEn.py output')

	# Overriding convert_arg_line_to_args in the input parser with our own function.
	parser.convert_arg_line_to_args = convert_arg_line_to_args

	parser.add_argument('--pdb',type=str,nargs=1,help='Path to the PDB file of the protein system')

	parser.add_argument('--resmeanintenfile',type=str,nargs=1,
		help='Path to the average interaction energy matrix produced by getResIntEn.py')

	parser.add_argument('--resintcorrfile',type=str,nargs=1,
		help='Residue interaction correlation list file produced by getResIntEn.py')

	parser.add_argument('--rescorrfile',type=str,nargs=1,
		help='Residue correlation matrix produced by getResIntEn.py')

	parser.add_argument('--includecovalents',action='store_true',default=False,
		help='Whether to include covalent bonds or not while constructing the network edges.')

	parser.add_argument('--intencutoff',type=float,nargs=1,default=[1],
		help='Mean (average) interaction energy cutoff when constructing the Ribeiro-Ortiz network \
		(kcal/mol). If an interaction energy time series absolute average value is below this \
		cutoff, not edge will be added between the two residues. By default, the cutoff is 1 kcal/mol.')

	parser.add_argument('--rescorrcutoff',type=float,nargs=1,default=[0.4],
		help='Residue correlation cutoff for inserting the edges between two residues in Kong-Karplus \
		network.')

	parser.add_argument('--outprefix',type=str,nargs=1,default=['energies_resNetwork_'],
		help='Output file name prefix')

	# Parse input arguments
	args = parser.parse_args()

	pdb = args.pdb[0]
	resMeanIntEnFile = args.resmeanintenfile[0]
	intEnCutoff = args.intencutoff[0]
	includeCovalents = args.includecovalents
	resIntCorrFile = args.resintcorrfile[0]
	resCorrCutoff = args.rescorrcutoff[0]
	resCorrFile = args.rescorrfile[0]
	outPrefix = args.outprefix[0]

	print('Constructing the Ribeiro-Ortiz network using average residue interaction energies.')
	getRibeiroOrtizNetwork(pdb=pdb,resMeanIntEnFile=resMeanIntEnFile,includeCovalents=includeCovalents,
		intEnCutoff=intEnCutoff,outName=outPrefix)

	print('Constructing the Kong-Karplus network using residue interaction energy correlations.')
	getKongKarplusNetwork(resIntCorrFile=resIntCorrFile,resCorrFile=resCorrFile,pdb=pdb,
		resMeanIntEnFile=resMeanIntEnFile,includeCovalents=includeCovalents,corrCutoff=resCorrCutoff,
		intEnCutoff=intEnCutoff,outName=outPrefix)
	print('Done. Please find your networks as gml files.')