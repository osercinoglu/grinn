#!/usr/bin/env python
import scipy.stats as stats
import numpy as np
import pyprind
import os
import itertools
import multiprocessing
import getResIntEn
import argparse
import datetime
import logging
import signal
from common import parseEnergiesSingleCore

def getResIntCorr(inFolder,logFile,frameRange=False,
	numCores=1,meanIntEnCutoff=float(1),outFile='resIntCorr.dat'):

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)
	logger.info('Started interaction energy correlation calculation.')

	# Get a list of interaction energy files in this folder.
	fileList = os.listdir(inFolder)
	fileList = [filename for filename in fileList if filename.endswith('energies.dat')]
	numInteractions = len(fileList)

	# For each file, determine whether the mean (average) value of the absolute interaction energy
	# is above the cutoff value. If it is above the value, include that file name in the filtered
	# list.
	filesFiltered = list()

	progBar = pyprind.ProgBar(numInteractions)
	for fileName in fileList:
		en = parseEnergiesSingleCore([inFolder+'/'+fileName])
		en = en.values()[0]['Total']
		mean_en = np.mean(np.abs(en))

		if mean_en > meanIntEnCutoff:
			filesFiltered.append(fileName)

		progBar.update()

	# For each file which passed the filtering step, generate pairwise combinations list and
	# compute pearson's correlation between each pairwise combination

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

	# Generate all possible unique combinations between filtered interaction energy files
	intCombins = list(itertools.combinations(filesFiltered,2))

	# Split this list into chunks according to the number of cores.
	intCombinsChunks = np.array_split(intCombins,numCores)

	# Start the correlation calculation in chunks
	enCorrResults = pool.map(getResIntCorrSingleCore,
		itertools.izip(intCombinsChunks,itertools.repeat(inFolder),
			itertools.repeat(logFile)))

	# Accumulate the output
	enCorrs = dict()
	for enCorrResult in enCorrResults:
		enCorrs = dict(enCorrs.items() + enCorrResult.items())

	pool.close()
	pool.join()

	# Write the results to the outFile 
	write2file(enCorrs,outFile)
	
	return enCorrs

def getResIntCorrSingleCore(args):

	intCombins = args[0]
	inFolder = args[1]
	logFile = args[2]
	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)
	
	numIntCombins = len(intCombins)
	enCorr = dict()

	progPercent = pyprind.ProgPercent(numIntCombins)

	logger.info('Started an interaction energy correlation calculation thread.')
	i = 0
	for intCombin in intCombins:
		en1 = getResIntEn.parseEnergiesSingleCore([inFolder+'/'+intCombin[0]])
		en1_keys = tuple(en1.keys())
		en1_values = en1.values()[0]['Total']
		en2 = getResIntEn.parseEnergiesSingleCore([inFolder+'/'+intCombin[1]])
		en2_keys = tuple(en2.keys())
		en2_values = en2.values()[0]['Total']
		pearson_r,_ = stats.pearsonr(en1_values,en2_values)
		enCorr[(en1_keys,en2_keys)] = pearson_r
		progPercent.update()
		i += 1
		percent = i/float(numIntCombins)
		logger.info('Interaction energy correlation thread calculated percentage: '+ str(percent*100))

	return enCorr

def write2file(enCorrs,outFile):

	f = open(outFile,'w')

	for key,value in enCorrs.iteritems():
		# Get the four residues involved in an interaction correlation.
		res1 = key[0][0][0]
		res2 = key[0][0][1]
		res3 = key[1][0][0]
		res4 = key[1][0][1]
		corr = value

		f.write('%i\t%i\t%i\t%i\t%f\n' % (res1,res2,res3,res4,corr))

	f.close()

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

	parser.add_argument('--infolder',type=str,nargs=1,help='Path to the folder where interaction\
		energies are located in')

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

	inFolder = args.infolder[0]
	numCores = args.numcores[0]
	meanIntEnCutoff = args.meanintencutoff[0]
	outFile = args.outfile[0]
	logFile = args.logfile[0]

	getResIntCorr(inFolder=inFolder,numCores=numCores,meanIntEnCutoff=meanIntEnCutoff,
		outFile=outFile,logFile=logFile)
