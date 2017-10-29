#!/usr/bin/env python3
from prody import *
import multiprocessing
import numpy as np
import sys, itertools, argparse, os, pyprind, subprocess
import re, pickle, types, logging, datetime, psutil, signal, time
import pandas
from getResIntEnMean import getResIntEnMean
from common import parseEnergiesSingleCore
import getResIntCorr

def filterPairsSingleCore(args):
	#SIDELINED (DEPRECATED)
	pairChunk = args[0]
	pdbPath = args[1]
	dcdPath = args[2]
	pairFilterBasis = args[3]
	pairFilterPercentage = args[4]
	pairFilterCutoff = args[5]
	skip = args[6]
	frameRange = args[7]
	logFile = args[8]

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started a filtering thread.')
	pdb = parsePDB(pdbPath)

	if type(frameRange) == types.BooleanType:
		traj = parseDCD(dcdPath,step=skip)
	elif len(frameRange) == 2:
		traj = parseDCD(dcdPath,step=skip,start=frameRange[0],stop=frameRange[1])	

	traj.setAtoms(pdb)
	numFrames = len(traj)

	pairsFiltered = list()

	progbar = pyprind.ProgBar(len(pairChunk))
	monitor = 0
	for pair in pairChunk:

		logFile = open('getResIntEn.log','w')
		pairDistances = np.zeros(numFrames)
		
		for i in xrange(0,numFrames):
			conformation = traj[i]
			atoms = conformation.getAtoms()
			atoms.setCoords(conformation.getCoords())
			
			if pairFilterBasis == 'com':
				res1atoms = atoms.select('resindex %i' % list(pair)[0])
				res2atoms = atoms.select('resindex %i' % list(pair)[1])
				res1atomsCOM = calcCenter(res1atoms,weights=res1atoms.getMasses())
				res2atomsCOM = calcCenter(res2atoms,weights=res2atoms.getMasses())

				distance = calcDistance(res1atomsCOM,res2atomsCOM)
				pairDistances[i] = distance

			elif pairFilterBasis == 'ca':
				res1CA = atoms.select('name CA and resindex %i' % list(pair)[0])
				res2CA = atoms.select('name CA and resindex %i' % list(pair)[1])

				distance = calcDistance(res1CA,res2CA)
				pairDistances[i] = distance

		# Further filter according to pairFilterPercentage.
		if not pairFilterPercentage:
			minPairDistance = np.min(pairDistances)

			if minPairDistance <= pairFilterCutoff:
				pairsFiltered.append(pair)

		elif pairFilterPercentage:
			#it is possible that it has str type, force it to be converted to a float.
			pairFilterPercentage = float(pairFilterPercentage)
			pairDistances = np.asarray(pairDistances)
			numAbovePercentage = len(np.where(pairDistances < pairFilterCutoff)[0])
			#print numAbovePercentage, numFrames, float(numAbovePercentage)/float(numFrames),type(pairFilterPercentage), pairFilterPercentage

			if float(numAbovePercentage)/float(len(pairDistances)) >= pairFilterPercentage:
				#print 'adding this pair'
				pairsFiltered.append(pair)

		progbar.update()

		monitor = monitor + 1

		calculatedPercentage = float(monitor)/float(len(pairChunk))*100
		logger.info('Filtered pairs percentage: %s' % str(calculatedPercentage))
		logFile.write('%s' % str(calculatedPercentage))
		logFile.close()

	logger.info('Completed a filtering thread.')
	return pairsFiltered

	sys.stdout.flush()

def calcEnergiesSingleCore(args):

	# Input arguments
	pairsFiltered = args[0]
	psfFilePath = args[1]
	dcdFilePath = args[2]
	skip = args[3]
	frameRange = args[4]

	# If frameRange is False, then the user did not request a frame range and thus
	# wants to include all frames in the analysis. In this case create an array 
	# for frameRange [0,-1] to indicate that we want all frames to the external tcl script
	if frameRange == False:
		frameRange = [0,-1]

	outputFolder = args[5]
	namd2exe = args[6]
	paramFile = args[7]
	logFile = args[8]

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started a pairwise energy calculation thread.')

	# Defining a method to calculate energies in chunks (to show the progress on the screen).
	def calcEnergiesSingleChunk(pairsFiltered,psfFilePath,dcdFilePath,skip,frameRange,
		outputFolder,namd2exe,paramFile,logger):

		# Construct a list of pairs (input argument string to the external tcl script)
		pairListArgConstruct = list()
		for pair in pairsFiltered:
			pair = list(pair)
			pairListArgConstruct.append(str(pair[0]))
			pairListArgConstruct.append(str(pair[1]))

		# Get the path of module, necessary when providing vmd the location of calcResIntEn.tcl.
		# (see below).
		module_path = sys.path[0]

		# Start a devnull for storing vmd/namd output.
		devnull = open(os.devnull,'w')
		
		vmdArgs = [namd2exe,outputFolder,psfFilePath,dcdFilePath,str(skip),
				str(frameRange[0]),str(frameRange[1]),str(paramFile)]
		
		print(module_path)
		vmdArgs = ['vmd','-dispdev','text','-e','%s/calcResIntEn.tcl' % module_path,'-args'] + vmdArgs + pairListArgConstruct
		
		pid_vmd = subprocess.Popen(vmdArgs,stdout=devnull)
		pid = os.getpid()
		#print(pid)
		logger.info('Started a pairwise energy calculation chunk with PID: %i' % pid)
		logger.info('Started a pairwise energy calculation chunk with VMD PID: %i' % pid_vmd.pid)

		pid_vmd.wait()

		logger.info('Completed a pairwise energy calculation chunk with PID: %i' % pid)
		logger.info('Completed a pairwise energy calculation chunk with VMD PID: %i' % pid_vmd.pid)

	# Split it into ten chunks to print the progress on the screen.
	pairsFilteredChunks = np.array_split(pairsFiltered,10)

	progBar = pyprind.ProgBar(10)

	# Perform the calculations in chunks
	percent = 0

	for pairsFilteredChunk in pairsFilteredChunks:
		calcEnergiesSingleChunk(pairsFilteredChunk,psfFilePath,dcdFilePath,skip,frameRange,
			outputFolder,namd2exe,paramFile,logger)

		progBar.update()
		percent = percent + 10
		logger.info('Completed pairwise interaction percentage: %s' % percent)

	logger.info('Completed a pairwise energy calculation thread.')

def getResIntEn(psf,pdb,dcd,numCores,sourceSel,targetSel,prePairCalc,prePairFilterCutoff,
	prePairFilterBasis,prePairFilterPercentage,prePairFilterSkip,pairCalc,pairFilterCutoff,
	pairFilterBasis,pairFilterPercentage,pairFilterSkip,skip,frameRange,outputFolder,namd2exe,paramFile,
	resIntCorr,resIntCorrAverageIntEnCutoff,toPickle,logFile):
	
	loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
	logging.basicConfig(format=loggingFormat,datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,
		filename=logFile)
	logger = logging.getLogger(__name__)
	
	# Also print messages to the terminal
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(logging.Formatter(loggingFormat))
	logger.addHandler(console)

	logger.info('Started calculation.')

	# ARGUMENT CHECKS
	if targetSel == False and pairCalc == True:
		logger.error("You must also specify a --targetsel for --paircalc to be executed,\
		 or vice versa. Aborting now.",exc_info=True)
		return

	# Start a log file (some crude logging functionality!)
	logFile2 = open('getResIntEn.log','w')
	logFile2.close()

	# Before we start the calculation, check whether the user has specified and outputFolder.
	# If yes, check whether it exists. If it exists, change to that directory and do calculations.
	# If it does not exist, create that directory and change to that directory and do calculations.
	currentFolder = os.getcwd()
	if outputFolder != currentFolder:
		if os.path.exists(outputFolder):
			logger.error(
				"The output folder exists. Please delete or rename this folder before"\
				 "proceeding. Aborting now.",exc_info=True)
			return
		if not os.path.isdir(outputFolder):
			logger.info('Creating the output folder %s' % outputFolder)
			os.makedirs(outputFolder)
			#os.chdir(outputFolder)

	if pairFilterCutoff < 4:
		logger.exception('Filtering distance cutoff value can not be smaller than 4. Aborting now.')

	try:
		system = parsePDB(pdb)
		systemCA = system.select('name CA')
	except:
		logger.exception('Could not load the PDB file provided. Please check your input PDB file.\
			 Aborting now.')
		return

	# Load psf with prody and get some useful numbers.
	### COMMENTING THIS OUT DUE TO INCOMPATIBILITY PROBLEMS WITH python3.6
	### WE DON'T NEED TO USE PSF IN PYTHON ANYWAY
	# try:
	# 	system = parsePSF(psf)
	# except:
	# 	logger.exception('Could not load the PSF file provided. Please check your input PDB file.\
	# 		 Aborting now.')
	# 	return

	try:
		traj = parseDCD(dcd)
	except:
		logger.exception('Could not load the DCD file provided. Please check your input DCD file.')
		return


	# Load pdb with prody and get some useful numbers.
	try:
		sourceCA = system.select(sourceSel+' and name CA')
	except:
		logger.exception('Could not select Selection 1 residue group. Aborting now.')
		return

	numSource = len(sourceCA)
	sourceResids = sourceCA.getResindices()
	sourceResnums = sourceCA.getResnums()
	sourceSegnames = sourceCA.getSegnames()

	allResiduesCA = system.select('name CA')
	numResidues = len(allResiduesCA)
	numTarget = numResidues

	# By default, targetResids are all residues.
	targetResids = np.arange(numResidues)
	
	# Get target selection residues, if provided:
	if targetSel:
		try:
			targetCA = system.select(targetSel+' and name CA')
		except:
			logger.exception('Could not select Selection 2 residue group. Aborting now.')
			return

		numTarget = len(targetCA)
		targetResids = targetCA.getResindices()

	# Generate all possible unique pairwise residue-residue combinations
	pairProduct = itertools.product(sourceResids,targetResids)
	pairSet = set()
	for x,y in pairProduct:
		if x != y:
			pairSet.add(frozenset((x,y)))

	# Split the pair set list into chunks according to number of cores
	pairChunks = np.array_split(list(pairSet),numCores)

	logger.info('Starting the filtering step.')

	# Continue with filtering operation
	print(sourceCA)
	traj.setAtoms(systemCA)
	coordSets = traj.getCoordsets()

	# Start a contact matrix (Kirchhoff matrix)
	kh = np.zeros((system.numResidues(),system.numResidues()))

	# Accumulate contact matrix as the sim progresses
	calculatedPercentage = 0
	monitor = 0
	for coordSet in coordSets:
		log = open('getResIntEn.log','w')
		gnm = GNM('GNM')
		gnm.buildKirchhoff(coordSet,cutoff=pairFilterCutoff)
		kh = kh + gnm.getKirchhoff()
		monitor = monitor + 1
		calculatedPercentage = (float(monitor)/float(len(coordSets)))*100
		log.write('%s' % str(calculatedPercentage))
		log.close()
		logger.info('Filtered pairs percentage: %s' % str(calculatedPercentage))

	# Get whether contacts are below cutoff for the specified percentage of simulation
	pairsFilteredFlag = np.abs(kh)/len(traj) > pairFilterCutoff*0.01

	pairsFiltered = list()
	#concatSourceTargetResids = np.concatenate([sourceResids,targetResids])
	for sourceResid in sourceResids:
		for targetResid in targetResids:
			if sourceResid == targetResid:
				continue
			elif pairsFilteredFlag[sourceResid,targetResid]:
				pairsFiltered.append(sorted([sourceResid,targetResid]))

	pairsFiltered = sorted(pairsFiltered)
	pairsFiltered = [list(x) for x in set(tuple(x) for x in pairsFiltered)]
	# file = open('pairsFiltered.txt','w')
	# for pair in pairsFiltered:
	# 	file.write('%i-%i\n' % (pair[0],pair[1]))
	# file.close()

	if not pairsFiltered:
		logger.exception('Filtering step did not yield any pairs. '
			'Either your cutoff value is too small or the percentage criteria is too high.')
		return

	logFile2= open('getResIntEn.log','w')
	logFile2.write('Number of interaction pairs selected after filtering step:\n')
	logFile2.write(str(len(pairsFiltered)))
	logFile2.close()

	# Start energy calculation in chunks
	pairsFilteredChunks = np.array_split(list(pairsFiltered),numCores)

	# Define a worker initializer for graceful exit upon ctrl+c
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

	pool.map(calcEnergiesSingleCore,
		zip(pairsFilteredChunks,itertools.repeat(psf),
			itertools.repeat(dcd),
			itertools.repeat(skip),
			itertools.repeat(frameRange),
			itertools.repeat(outputFolder),
			itertools.repeat(namd2exe),
			itertools.repeat(paramFile),
			itertools.repeat(logFile)))
	
	#pool.join()
	# Parse the specified outFolder after energy calculation is done.
	outFolderFileList = os.listdir(outputFolder)

	energiesFilePaths = list()
	for fileName in outFolderFileList:
		if fileName.endswith('energies.dat'):
			energiesFilePaths.append(outputFolder+'/'+fileName)

	energiesFilePathsChunks = np.array_split(list(energiesFilePaths),numCores)

	parsedEnergiesResults = pool.map(parseEnergiesSingleCore,energiesFilePathsChunks)
	
	# while not parsedEnergiesResults.ready():
	# 	print("num left: {}".format(parsedEnergiesResults._number_left))
	# 	time.sleep(1)

	# parsedEnergiesResults = parsedEnergiesResults.get()
	
	parsedEnergies = dict()
	for parsedEnergiesResult in parsedEnergiesResults:
		parsedEnergies.update(parsedEnergiesResult)

	# Prepare a pandas data table from parsed energies, write it to new files depending on type of energy
	df_total = pandas.DataFrame()
	df_elec = pandas.DataFrame()
	df_vdw = pandas.DataFrame()
	for key,value in list(parsedEnergies.items()):
		df_total[key] = value['Total']
		df_elec[key] = value['Elec']
		df_vdw[key] = value['VdW']

	df_total.to_csv(outputFolder+'/energies_intEnTotal.csv')
	df_elec.to_csv(outputFolder+'/energies_intEnElec.csv')
	df_vdw.to_csv(outputFolder+'/energies_intEnVdW.csv')

	# If saving to a pickle is requested:
	if toPickle:
		file = open(outputFolder+'.pickle','wb')
		pickle.dump(parsedEnergies,file)
		file.close()

	# Save average interaction energies as well!
	if toPickle:
		getResIntEnMean(outputFolder+'.pickle',pdb,prefix=outputFolder+'/energies')

	pool.close()
	pool.join()

	if resIntCorr:
		getResIntCorr.getResIntCorr(inFolder=outputFolder,pdb=pdb,numCores=numCores,meanIntEnCutoff=resIntCorrAverageIntEnCutoff,
			outFile=outputFolder+'/energies_IntEnCorr.dat',logFile=logFile)

	# Delete all namd-generated energies file from output folder.
	subprocess.call('rm %s/*_energies.dat' % outputFolder,shell=True)
	
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
		description='Calculate totalenergy of individual residues \
		of a protein or calculate pairwise nonbonded interaction energies between residues of two \
		selected groups of atoms of a protein over the course of a molecular dynamics DCD trajectory.\
		\n\n\
		getResidueEnergies requires VMD and NAMD to be installed and availabe for call from a \
		terminal window. For calculation of energies, it employs the namdenergy.tcl script from VMD/NAMD \
		developers.')

	# Overriding convert_arg_line_to_args in the input parser with our own function.
	parser.convert_arg_line_to_args = convert_arg_line_to_args

	parser.add_argument('--pdb',type=str,nargs=1,help='Name of the corresponding pdb file of the \
		DCD trajectory')

	parser.add_argument('--psf',type=str,nargs=1,help='Name of the corresponding psf file of the \
		DCD trajectory')

	parser.add_argument('--dcd',type=str,nargs=1,help='Name of the trajectory DCD file')

	parser.add_argument('--numcores',default=[1],type=int,nargs=1,
		help='Number of CPU cores to be employed for energy calculation. If not specified, it \
		defaults to 1 (no parallel computation will be done). If specified, e.g. NUMCORES=n, \
		then the computational workloading will be distributed among n cores.')

	parser.add_argument('--sourcesel',default=['all'],nargs='+',help='A ProDy atom selection \
	 string which determines the first group of selected residues. ')

	parser.add_argument('--targetsel',default=False,type=str,nargs='+',help='A ProDy atom selection string \
		which determines the second group of selected residues.')

	parser.add_argument('--prepaircalc',action='store_true',default=True,help='When given, a preliminary filtering operation is\
		done to yield residues for further energy evaluation.')

	parser.add_argument('--prepairfiltercutoff',type=float,default=[20],nargs=1,help='Cutoff distance (angstroms) \
		for preliminary filtering of pairwise amino acids. If not specified, it defaults to 20. \
		Only those residues that are within the PREPAIRFILTERCUTOFF distance of each other for at least once \
		throughout the trajectory will the taken into account for further evaluation.')

	parser.add_argument('--prepairfilterbasis',type=str,default=['com'],nargs=1,help='Basis for filtering of residues\
		pairs. It not specified, it defaults to "com" (residue center of mass). Possible values: "com", "ca"')

	parser.add_argument('--prepairfilterpercentage',type=float,default=[False],nargs=1,help='When given, residues that\
		are within the PREPAIRFILTERCUTOFF distance from each other for at least PREPAIRFILTERPERCENTAGE percent of \
		the trajectory will be taken into account in further evaluations. When not given, it defaults to False \
		(residues that are within PREPAIRFILTERPERCENTAGE from each other for at least once are taken).')

	parser.add_argument('--prepairfilterskip',type=int,default=[1],nargs=1,help='If specified, only PREPAIRFILTERSKIPth\
		frame will be evaluated during the preliminary filtering stage.')

	parser.add_argument('--paircalc',action='store_true',default=True,help='When given, this argument enables pairwise \
		nonbonded interaction energy calculation between sourcesel and targetsel residues. When not given, \
		total energy of each residue in sourcesel will be calculated.')

	parser.add_argument('--pairfiltercutoff',type=float,default=[15],nargs=1,help='Cutoff distance (angstroms) \
		for pairwise interaction energy calculations. If not specified, it defaults to 15. \
		Only those residues that are within the PAIRFILTERCUTOFF distance of each other for at least PAIRCUTOFFPERCENTAGE of\
		the trajectory will be taken into account in energy calculations.')

	parser.add_argument('--pairfilterbasis',type=str,default=['com'],nargs=1,help='Basis for filtering of residues\
		pairs. It not specified, it defaults to "com" (residue center of mass). Possible values: "com", "ca"')

	parser.add_argument('--pairfilterpercentage',default=[False],nargs=1,help='When given, residues that\
		are within the PAIRFILTERCUTOFF distance from each other for at least PAIRFILTERPERCENTAGE percent of \
		the trajectory will be taken into account in further evaluations. When not given, it defaults to 0.75 \
		(75%%) (residues that are within PREPAIRFILTERPERCENTAGE from each other for at least once are taken).')

	parser.add_argument('--pairfilterskip',type=int,default=[1],nargs=1,help='If specified, only PAIRFILTERSKIPth\
		frame will be evaluated during the filtering stage.')

	parser.add_argument('--skip',default=[1],type=int,nargs=1,help='If specified, namdenergy.tcl \
		will use this skip parameter, which defines the number of frames in dcd to skip between \
		each calculation.')

	parser.add_argument('--framerange',type=int,default=[False],nargs='+',help='If specified, then only FRAMERANGE\
		section of the trajectory will be handled')

	parser.add_argument('--namd2exe',default=['namd2'],type=str,nargs=1,help='Path to the namd2 executable.')

	parser.add_argument('--parameterfile',default=[False],type=str,nargs=1,help='Path to the parameter file.')

	parser.add_argument('--resintcorr',action='store_true',default=False,help='When True, interaction energy correlation \
		is also calculated following interaction energy calculation')

	parser.add_argument('--resintcorraverageintencutoff',default=[1],type=float,nargs=1,help='\
		Mean (average) interaction energy cutoff for filtering interaction energies \
		(kcal/mol). If an interaction energy time series absolute average value is below this \
		cutoff, that interaction energy will not be taken in correlation calculations.\
		By default, the cutoff is 1 kcal/mol.')

	parser.add_argument('--outfolder',default=[os.getcwd()],type=str,nargs=1,help='Folder path for storing \
		calculation results. If not specified, the current working folder will be used.')

	parser.add_argument('--topickle',default=False,action='store_true',help='When given, the energy values \
		are stored into a pickle file in the current working directory for later import and analysis in python.')

	now = datetime.datetime.now()
	logFile = 'getResIntEnLog_%4d%2d%2d_%2d%2d%2d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)
	parser.add_argument('--logfile',default=[logFile],type=str,nargs=1,help='Log file name')

	# Parsing input arguments
	args = parser.parse_args()

	psf = args.psf[0]
	pdb = args.pdb[0]
	dcd = args.dcd[0]
	
	numCores = args.numcores[0]
	frameRange = args.framerange
	skip = args.skip[0]

	prePairCalc = args.prepaircalc
	prePairFilterCutoff = args.prepairfiltercutoff[0]
	prePairFilterBasis = args.prepairfilterbasis[0]
	prePairFilterPercentage = args.prepairfilterpercentage[0]
	prePairFilterSkip = args.prepairfilterskip[0]

	pairCalc = args.paircalc
	pairFilterCutoff = args.pairfiltercutoff[0]
	pairFilterBasis = args.pairfilterbasis[0]
	pairFilterPercentage = args.pairfilterpercentage[0]
	pairFilterSkip = args.pairfilterskip[0]

	outputFolder = args.outfolder[0]

	namd2exe = args.namd2exe[0]

	paramFile = args.parameterfile[0]
	if not paramFile:
		paramFile = False

	logFile = args.logfile[0]

	if len(args.sourcesel) > 1:
		sourceSel = ' '.join(args.sourcesel)
	else:
		sourceSel = args.sourcesel[0]

	if args.targetsel:
		targetSel = ' '.join(args.targetsel)
	else:
		targetSel = False

	if len(args.framerange) > 1:
		frameRange = np.asarray(args.framerange)
	else:
		frameRange = args.framerange[0]

	resIntCorr = args.resintcorr 
	resIntCorrAverageIntEnCutoff = args.resintcorraverageintencutoff[0]

	getResIntEn(psf=psf,pdb=pdb,dcd=dcd,numCores=numCores,
		sourceSel=sourceSel,targetSel=targetSel,prePairCalc=prePairCalc,
		prePairFilterCutoff=prePairFilterCutoff,prePairFilterBasis=prePairFilterBasis,
		prePairFilterPercentage=prePairFilterPercentage,prePairFilterSkip=prePairFilterSkip,
		pairCalc=pairCalc,pairFilterCutoff=pairFilterCutoff,pairFilterBasis=pairFilterBasis,
		pairFilterPercentage=pairFilterPercentage,pairFilterSkip=pairFilterSkip,
		skip=skip,frameRange=frameRange,resIntCorr=resIntCorr,
		resIntCorrAverageIntEnCutoff=resIntCorrAverageIntEnCutoff,outputFolder=outputFolder,
		namd2exe=namd2exe,paramFile=paramFile,toPickle=args.topickle,logFile=logFile)