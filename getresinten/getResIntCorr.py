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
import pyximport
import pandas
import psutil
pyximport.install()
import getResCorr
import re
from prody import *
from common import parseEnergiesSingleCore

def getResIntCorrSingleCore(args):

	e_combins = args[0]
	sliceIndices = args[1]
	# Generate all possible unique combinations between filtered interaction energy files
	intCombins = itertools.combinations(e_combins,2)
	intCombins_slice = itertools.islice(intCombins,sliceIndices[0],sliceIndices[1])
	df_energies = args[2]
	logFile = args[3]
	corrCutoff = 0.4

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Starting interaction energy correlation calculation thread...')
	# Get pearson correlation between all energy combinations
	progbar = pyprind.ProgBar(sliceIndices[1]-sliceIndices[0],monitor=True)

	#e_corrs = np.zeros([len(e_combins),5],dtype=np.float)
	e_corrs = list()

	for combin in intCombins_slice:
		
		combin = combin[0]
		matches1 = re.search('(\d+), (\d+)',combin[0])
		matches2 = re.search('(\d+), (\d+)',combin[1])
		combin1 = list(map(int,[matches1.groups()[0],matches1.groups()[1]]))
		combin2 = list(map(int,[matches2.groups()[0],matches2.groups()[1]]))
		i = combin1[0]
		j = combin1[1]
		k = combin2[0]
		l = combin2[1]
		
		pearson_r,_ = stats.pearsonr(df_energies[combin[0]].values,df_energies[combin[1]].values)
		
		if pearson_r >= corrCutoff:
			e_corrs.append([i,j,k,l,pearson_r])

		progbar.update()

	logger.info('Starting interaction energy correlation calculation thread... completed.')

	return e_corrs
	
def write2file(enCorrs,outFile):

	f = open(outFile,'w')

	for key,value in enCorrs.items():
		# Get the four residues involved in an interaction correlation.
		res1 = key[0][0][0]
		res2 = key[0][0][1]
		res3 = key[1][0][0]
		res4 = key[1][0][1]
		corr = value

		f.write('%i\t%i\t%i\t%i\t%f\n' % (res1,res2,res3,res4,corr))

	f.close()

def getResIntCorr(inFile,pdb,logFile,frameRange=False,
	numCores=1,meanIntEnCutoff=float(1),outFile='resIntCorr.dat'):

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)
	logger.info('Started residue interaction energy calculation.')

	# Load previously computed interaction energy values (from previously saved pickle files)
	logger.info('Reading input CSV...')
	df_energies = pandas.read_csv(inFile)
	logger.info('Reading input CSV... completed.')

	numResPairs = len(df_energies.columns[1:]) # Beware: first column is the frame

	# Take absolute average interaction energy pairs above energyCutoff
	logger.info('Filtering out interaction below energy threshold...')
	progbar = pyprind.ProgBar(numResPairs,monitor=True)

	valid_e = list()
	for column in df_energies.columns[1:]:
		if np.abs(np.mean(df_energies[column].values)) > meanIntEnCutoff:
			if natsorted(column) not in valid_e:
				valid_e.append(column)
		progbar.update()

	logger.info('Filtering out interaction below energy threshold... completed.')
	logger.info('A total of %i interaction energies will enter interaction energy correlation calculation.' 
		% len(valid_e))

	logger.info('Getting dual combinations betweel all interaction energy pairs...')
	# Get dual combinations between these interaction energy pairs
	e_combins = list(itertools.combinations(valid_e,2))
	logger.info('Getting dual combinations betweel all interaction energy pairs... completed.')
	logger.info('A total of %i interaction energy correlations will be computed.' % len(e_combins))
	logger.info('Starting interaction energy correlation calculation...')

	# Get the number of possible unique combinations between filtered interacton energy files

	numIntCombins = len(e_combins)**2

	logger.info('Splitting interaction energy correlation dual combinations into chunks...')
	# Generate start stop indices for each chunk in iterator
	stepIntCombins = math.ceil(numIntCombins/24)

	sliceIndex = 0
	sliceIndices = list()
	sliceIndices.append(sliceIndex)
	for i in range(0,numCores):
		sliceIndex += stepIntCombins
		sliceIndices.append(sliceIndex)

	sliceIndices[-1] = numIntCombins
	# Split this list into chunks according to the number of cores.
	sliceIndexChunks = [[sliceIndices[i],sliceIndices[i+1]] for i in range(0,len(sliceIndices)-1)]

	logger.info('Splitting interaction energy correlation dual combinations into chunks... completed.')

	parent_id = os.getpid()
	def worker_init():
		def sig_int(signal_num, frame):
			print('signal: %s' % signal_num)
			parent = psutil.Process(parent_id)
			for child in parent.children():
				if child.pid != os.getpid():
					print("killing child: %s" % child.pid)
					child.kill()
			print("killing parent: %s" % parent_id)
			parent.kill()
			print("suicide: %s" % os.getpid())
			psutil.Process(os.getpid()).kill()
		signal.signal(signal.SIGINT, sig_int)

	# Start a pool of processors
	pool = multiprocessing.Pool(numCores,worker_init)

	# Start the correlation calculation in chunks
	e_corrsMap = pool.map(getResIntCorrSingleCore,
		zip(itertools.repeat(e_combins),sliceIndexChunks,itertools.repeat(df_energies),
			itertools.repeat(logFile)))

	e_corrs = list()
	for result in e_corrsMap:
		e_corrs += result

	pool.close()
	pool.join()

	del df_energies # Save RAM
	del intCombins # Save RAM

	raise SystemExit(0)
	# Write the results to the outFile 
	#write2file(enCorrs,outFile)

	# Calculate residue correlation matrix (based on Kong & Karplus, 2008)
	# as well.
	getResCorr.getResCorr(e_corrs,pdb,corrCutoff=0.4,outName=outFile,logFile=logFile)
	
	return e_corrs

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

	parser.add_argument('--numcores',type=int,nargs=1,default=[1],help='Number of CPU cores to be\
		employed for energy calculation. If not specified, it defaults to 1 (no parallel \
		computation will be done). If specified, e.g. NUMCORES=n, then the computational \
		workloading will be distributed among n cores.')

	parser.add_argument('--meanintencutoff',type=float,nargs=1,default=[1],
		help='Mean (average) interaction energy cutoff for filtering interaction energies \
		(kcal/mol). If an interaction energy time series absolute average value is below this \
		cutoff, that interaction energy will not be taken in correlation calculations.\
		By default, the cutoff is 1 kcal/mol.')

	parser.add_argument('--outfile',type=str,nargs=1,default=['resIntCorr.dat'],
		help='Path of the file for storing calculation results. If not specified, the default value\
		is resIntCorr.dat in the current working directory')

	now = datetime.datetime.now()
	logFile = 'getResIntCorrLog_%d%d%d_%d%d%d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)
	parser.add_argument('--logfile',default=[logFile],type=str,nargs=1,help='Log file name')

	# Parse input arguments
	args = parser.parse_args()

	inFolder = args.infile[0]
	pdb = args.pdb[0]
	numCores = args.numcores[0]
	meanIntEnCutoff = args.meanintencutoff[0]
	outFile = args.outfile[0]
	logFile = args.logfile[0]

	getResIntCorr(inFile=inFolder,pdb=pdb,numCores=numCores,meanIntEnCutoff=meanIntEnCutoff,
		outFile=outFile,logFile=logFile)
