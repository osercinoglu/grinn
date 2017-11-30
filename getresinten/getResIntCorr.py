#!/usr/bin/env python3
import scipy.stats as stats
from natsort import natsorted
import numpy as np
import pyprind
import os
import math
import itertools
import multiprocessing
import getResIntEn
import argparse
import datetime
import logging
import signal
import pandas
import psutil
import re
from prody import *
from common import parseEnergiesSingleCoreNAMD
from common import getChainResnameResnum

def getResIntCorr(inFile,pdb,logFile=None,logger=None,frameRange=False,
	numCores=1,meanIntEnCutoff=float(1),outPrefix=''):
	
	if not logger and logFile:
		logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
		logger = logging.getLogger(__name__)
	
	logger.info('Started residue interaction energy calculation.')

	logger.info('Reading input CSV...')
	# Read in interaction energy time series from the getResIntEn csv output.
	df = pandas.read_csv(inFile,nrows=10)
	logger.info('Reading input CSV... completed.')

	# Get number of residues
	system = parsePDB(pdb)
	systemProtein = system.select('protein or nucleic')
	numResidues = len(np.unique(systemProtein.getResindices()))

	# Convert the interaction energy time series to a 3D matrix
	intEnMat = np.zeros((numResidues,numResidues,len(df)))

	logger.info('Calculating interaction energy correlations...')
	for i in range(0,numResidues):
		for j in range(0,numResidues):
			df_col1 = getChainResnameResnum(system,i)+'-'+getChainResnameResnum(system,j)
			df_col2 = getChainResnameResnum(system,j)+'-'+getChainResnameResnum(system,i)
			if df_col1 in df.columns:
				df_col = df_col1
			elif df_col2 in df.columns:
				df_col = df_col2
			else:
				intEnMat[i,j] = np.zeros(len(df))
				intEnMat[j,i] = np.zeros(len(df))
				continue

			# Only take interactions whose average is above meanIntEnCutoff
			if np.mean(df[df_col].values) >= meanIntEnCutoff:
				intEnMat[i,j] = df[df_col1].values
				intEnMat[j,i] = df[df_col1].values
			else:
				intEnMat[i,j] = np.zeros(len(df))
				intEnMat[j,i] = np.zeros(len(df))

		percentCalculated = ((i+1)/float(numResidues))*100/2 # /2 because this is only halfway of calculation.
		logger.info('Interaction energy correlation calculated percentage: %f' % percentCalculated)


	# Calculate pearson product moment correlation between all interaction energy pairs
	# using linear algebra (matrix formalism)
	# This is much more efficient than using for loops

	progbar = pyprind.ProgBar(numResidues)

	# Store correlations in a dictionary.
	sigcorrs = dict()
	for i in range(0,numResidues):
		# First get all correlations between interactions involving residue i
		row = intEnMat[i,:]
		row_mrow = row - row.mean(1)[:,None]
		ssrow = (row_mrow**2).sum(1);
		corrs = np.dot(row_mrow,row_mrow.T)/np.sqrt(np.dot(ssrow[:,None],ssrow[None]))
		sigindices = np.where(corrs > 0.4)
		for m in range(0,len(sigindices[0])):
			row = sigindices[0][m]
			col = sigindices[1][m]
			if row != col and i != row and i != col:
				key = '-'.join(list(map(str,natsorted([i,row])+natsorted([i,col]))))
				if key not in list(sigcorrs.keys()):
					sigcorrs[key] = corrs[row,col]
		for j in range(0,numResidues):
			# Now get all correlations involving residue i and j
			row2 = intEnMat[j,:]
			row2_mrow = row2 - row2.mean(1)[:,None]
			ssrow2 = (row2_mrow**2).sum(1);
			corrs = np.dot(row_mrow,row2_mrow.T)/np.sqrt(np.dot(ssrow[:,None],ssrow2[None]))
			sigindices = np.where(corrs > 0.4)
			for m in range(0,len(sigindices[0])):
				row = sigindices[0][m]
				col = sigindices[1][m]
				if row != col and i != row and j != col and not (i == col and j == row): # Excluding correlations with self.
					key = '-'.join(list(map(str,natsorted([i,row])+natsorted([j,col]))))
					if key not in list(sigcorrs.keys()):
						sigcorrs[key] = corrs[row,col]

		percentCalculated = 50+((i+1)/float(numResidues))*100/2 # /2 because this is only halfway of calculation.
		logger.info('Interaction energy correlation calculated percentage: %f' % percentCalculated)

		progbar.update()

	logger.info('Calculating interaction energy correlations... completed.')

	logger.info('Saving correlations to file...')
	df_corr = pandas.DataFrame(columns=['res11','res12','res21','res22','corr'])

	for i in range(0,len(sigcorrs)):
		key = list(sigcorrs.keys())[i]
		matches = re.search('(\d+)-(\d+)-(\d+)-(\d+)',key)
		res11 = int(matches.groups()[0])
		res12 = int(matches.groups()[1])
		res21 = int(matches.groups()[2])
		res22 = int(matches.groups()[3])

		res11_string = getChainResnameResnum(system,res11)
		res12_string = getChainResnameResnum(system,res12)
		res21_string = getChainResnameResnum(system,res21)
		res22_string = getChainResnameResnum(system,res22)

		# Do not include correlations with self.
		#if res11_string == res22_string and res12_string == res21_string:
		#	continue

		df_corr.loc[i] = [res11_string,res12_string,res21_string,res22_string,sigcorrs[key]]

	df_corr.to_csv(outPrefix+'_resIntCorr.csv')

	# Constructing the residue correlation matrix
	logger.info('Constructing the residue correlation matrix...')
	rc = np.zeros((numResidues,numResidues))

	progbar = pyprind.ProgBar(len(sigcorrs))
	for key in list(sigcorrs.keys()):
		matches = re.search('(\d+)-(\d+)-(\d+)-(\d+)',key)
		rc_key = np.zeros((numResidues,numResidues))
		if matches:
			res11 = int(matches.groups()[0])
			res12 = int(matches.groups()[1])
			res21 = int(matches.groups()[2])
			res22 = int(matches.groups()[3])

			corr = np.abs(sigcorrs[key])

			rc_key[res11,res21] = corr
			rc_key[res11,res22] = corr
			rc_key[res12,res21] = corr
			rc_key[res12,res22] = corr
			rc_key[res21,res11] = corr
			rc_key[res21,res12] = corr
			rc_key[res22,res11] = corr
			rc_key[res22,res12] = corr

			rc = rc + rc_key

		progbar.update()

	logger.info('Constructing the residue correlation matrix... completed.')

	np.savetxt(outPrefix+'_resCorr.dat',rc)


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
		description='Calculate pearson correlation factor between residue-residue interactions \
		calculated by getResIntEn.py')

	# Overriding convert_arg_line_to_args in the input parser with our own function.
	parser.convert_arg_line_to_args = convert_arg_line_to_args

	parser.add_argument('--infile',type=str,nargs=1,help='Path to the CSV file where interaction\
		energies are located in')

	parser.add_argument('--pdb',type=str,nargs=1,help='Path to the PDB file of the protein system')

	parser.add_argument('--meanintencutoff',type=float,nargs=1,default=[1],
		help='Mean (average) interaction energy cutoff for filtering interaction energies \
		(kcal/mol). If an interaction energy time series absolute average value is below this \
		cutoff, that interaction energy will not be taken in correlation calculations.\
		By default, the cutoff is 1 kcal/mol.')

	parser.add_argument('--outprefix',type=str,nargs=1,default=[''],
		help='Path of the file for storing calculation results. If not specified, the default value\
		is resIntCorr.dat in the current working directory')

	now = datetime.datetime.now()
	logFile = 'getResIntCorrLog_%d%d%d_%d%d%d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)
	parser.add_argument('--logfile',default=[logFile],type=str,nargs=1,help='Log file name')

	# Parse input arguments
	args = parser.parse_args()

	inFile = args.infile[0]
	pdb = args.pdb[0]
	meanIntEnCutoff = args.meanintencutoff[0]
	outPrefix = args.outprefix[0]
	logFile = args.logfile[0]

	getResIntCorr(inFile=inFile,pdb=pdb,meanIntEnCutoff=meanIntEnCutoff,
		outPrefix=outPrefix,logFile=logFile)
