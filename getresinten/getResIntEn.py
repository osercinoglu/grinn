#!/usr/bin/env python3
from prody import *
import mdtraj
import multiprocessing
import pexpect
import numpy as np
import sys, itertools, argparse, os, pyprind, subprocess
import re, pickle, types, logging, datetime, psutil, signal, time
import pandas, glob
from shutil import copyfile
from getResIntEnMean import getResIntEnMean
from common import parseEnergiesSingleCoreNAMD
from common import getChainResnameResnum
from common import makeNDXMDPforGMX
from common import parseEnergiesGMX
import getResIntCorr

def calcEnergiesSingleCoreNAMD(args):
	# Input arguments
	pairsFiltered = args[0]
	psfFilePath = args[1]
	pdbFilePath = args[2]
	dcdFilePath = args[3]
	skip = args[4]
	frameRange = args[5]
	pairFilterCutoff = args[6]
	environment = args[7]
	soluteDielectric = args[8]
	solventDielectric = args[9]

	# If frameRange is False, then the user did not request a frame range and thus
	# wants to include all frames in the analysis. In this case create an array 
	# for frameRange [0,-1] to indicate that we want all frames to the external tcl script
	if frameRange == False:
		frameRange = [0,-1]

	outputFolder = args[10]
	namd2exe = args[11]
	paramFile = args[12]
	logFile = args[13]

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started an energy calculation thread.')

	# Defining a method to calculate energies in chunks (to show the progress on the screen).
	def calcEnergiesSingleChunk(pairsFiltered,psfFilePath,pdbFilePath,dcdFilePath,skip,frameRange,
		pairFilterCutoff,environment,soluteDielectric,solventDielectric,outputFolder,namd2exe,paramFile,
		logger):
		for pair in pairsFiltered:
			# Write PDB files for pairInteractionGroup specification
			system = parsePDB(pdbFilePath)
			sel1 = system.select('resindex %i' % int(pair[0]))
			sel2 = system.select('resindex %i' % int(pair[1]))
			# Changing the values of B-factor columns so that they can be recognized by
			# pairInteractionGroup1 parameter in NAMD configuration file.
			sel1.setBetas([1]*sel1.numAtoms())
			sel2.setBetas([2]*sel2.numAtoms())
			pairIntPDB = '%s/%i_%i-temp.pdb' % (outputFolder,pair[0],pair[1])
			pairIntPDB = os.path.abspath(pairIntPDB)
			writePDB(pairIntPDB,system)

			# SAVING ON THE TWO RESIDUE PAIR TO DO LATER ON(NEEDS TESTING)
			#traj = Trajectory(dcdFilePath)
			#traj.link(system)
			
			#traj.setAtoms(system.select('resindex %i %i' % (pair[0],pair[1])))
			#writeDCD('%i_%i-temp.dcd' % (pair[0],pair[1]),traj)
			
			namdConf = '%s/%s_%s-temp.namd' % (outputFolder,pair[0],pair[1])
			f = open(namdConf,'w')

			f.write('structure %s\n' % psfFilePath)
			f.write('paraTypeCharmm on\n')
			if paramFile:
				for file in paramFile:
					#raise SystemExit(0)
					f.write('parameters %s\n' % file)
			else:
				f.write('parameters %s\n' % (sys.path[0]+'/par_all27_prot_lipid_na.inp'))
			f.write('numsteps 1\n')
			f.write('switching off\n')
			f.write('exclude scaled1-4\n')
			f.write('outputname %i_%i-temp\n' % (pair[0],pair[1]))
			f.write('temperature 0\n')
			f.write('COMmotion yes\n')
			f.write('cutoff %d\n' % pairFilterCutoff)
			
			if environment == 'implicit-solvent':
				f.write('GBIS on\n')
				f.write('solventDielectric %d\n' % solventDielectric)
				f.write('dielectric %d\n' % soluteDielectric)
				f.write('alphaCutoff %d\n' % (float(pairFilterCutoff)-3)) # Setting GB radius to cutoff for now. We might want to change this behaviour later.
				f.write('SASA on\n')
			elif environment == 'vacuum':
				f.write('dielectric %d\n' % soluteDielectric)
			else:
				f.write('#environment is %s\n' % str(environment))

			f.write('switchdist 10.0\n')
			f.write('pairInteraction on\n')
			f.write('pairInteractionGroup1 1\n')
			f.write('pairInteractionFile %s\n' % pairIntPDB)
			f.write('pairInteractionGroup2 2\n')
			f.write('coordinates %s\n' % pairIntPDB)
			f.write('set ts 0\n')
			#f.write('coorfile open dcd %i_%i-temp.dcd\n' % (pair[0],pair[1]))
			f.write('coorfile open dcd %s\n' % dcdFilePath)
			f.write('while { ![coorfile read] } {\n')
			f.write('\tfirstTimeStep $ts\n')
			f.write('\trun 0\n')
			f.write('\tincr ts 1\n')
			for i in range(0,skip-1,1):
			 	f.write('\tcoorfile skip\n')
			f.write('}\n')
			f.write('coorfile close')
			f.close()

			# Run namd2 to compute the energies
			pid_namd2 = subprocess.Popen([namd2exe,namdConf],
				stdout=open(outputFolder+'/%i_%i_energies.log' % (pair[0],pair[1]),'w'))
			pid_namd2.wait()

			#subprocess.call('rm %s' % namdConf,shell=True)
			#subprocess.call('rm %s' % pairIntPDB,shell=True)
			#subprocess.call('rm %i_%i-temp*' % (pair[0],pair[1]),shell=True)
			#raise SystemExit(0)
		# Parse the log file and extract necessary energy values

		# Done.

	# Split it into ten chunks to print the progress on the screen.
	pairsFilteredChunks = np.array_split(list(pairsFiltered),10)

	progBar = pyprind.ProgBar(10)

	# Perform the calculations in chunks
	percent = 0

	for pairsFilteredChunk in pairsFilteredChunks:
		calcEnergiesSingleChunk(pairsFilteredChunk,psfFilePath,pdbFilePath,dcdFilePath,skip,frameRange,
			pairFilterCutoff,environment,soluteDielectric,solventDielectric,outputFolder,namd2exe,paramFile,logger)

		progBar.update()
		percent = percent + 10
		logger.info('Completed calculation percentage: %s' % percent)

	logger.info('Completed a pairwise energy calculation thread.')

def calcEnergiesGMX(pairsFiltered,topFilePath,pdbFilePath,tprFilePath,trajFilePath,skip,frameRange,
	pairFilterCutoff,outputFolder,gmxExe,logFile,numCores):
	
	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started an energy calculation thread.')

	# Prevent backup making while calculating energies.
	os.environ["GMX_MAXBACKUP"] = "-1"

	# Make an index and MDP file with the pairs filtered.
	#gmxExe = 'gmx'
	makeNDXMDPforGMX(gmxExe=gmxExe,pdb=pdbFilePath,tpr=tprFilePath,pairsFiltered=pairsFiltered,outFolder=outputFolder)

	# Call gromacs pre-processor (grompp) and make a new TPR file for each pair and calculate energies for each pair.
	i = 0
	for pair in pairsFiltered:
		proc = subprocess.Popen([gmxExe,'grompp','-f',outputFolder+'/interact'+str(i)+'.mdp','-n',
			outputFolder+'/interact.ndx','-p',topFilePath,'-c',tprFilePath,'-o',outputFolder+'/interact'+str(i)+'.tpr','-maxwarn','20'],
			stderr=subprocess.STDOUT,stdout = subprocess.PIPE)
		proc.wait()

		proc = subprocess.Popen([gmxExe,'mdrun','-rerun',trajFilePath,'-s',outputFolder+'/interact'+str(i)+'.tpr',
			'-e',outputFolder+'/interact'+str(i)+'.edr','-nt',str(numCores)],
			stderr=subprocess.STDOUT,stdout = subprocess.PIPE)
		proc.wait()

		i += 1

		logger.info('Completed calculation percentage: '+str(i/len(pairsFiltered)*100))

def getResIntEn(top,pdb,tpr,traj,numCores,sourceSel,targetSel,environment,soluteDielectric,solventDielectric,
	pairCalc,pairFilterCutoff,pairFilterBasis,pairFilterPercentage,pairFilterSkip,skip,frameRange,
	outputFolder,namd2exe,gmxExe,paramFile,resIntCorr,resIntCorrAverageIntEnCutoff,toPickle,logFile):
	
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

	# Registering signal handler to capture SIGINT.
	# def signal_handler(signal, frame):
	# 	print('Captured SIGINT')
	# 	os._exit(0)

	# signal.signal(signal.SIGINT, signal_handler)

	# ARGUMENT CHECKS
	# TEMP
	pairFilterSkip = skip

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

	if tpr:
		#gmxExe = 'gmx' # TEMPORARY

		# Convert tpr to pdb, selecting just Protein.
		proc = pexpect.spawnu('%s trjconv -f %s -s %s -b 0 -e 0 -o %s' % (gmxExe,traj,tpr,outputFolder+'/system_dry.pdb'))
		proc.send('Protein')
		proc.sendline()
		proc.wait()
		proc.kill(1)

		# Convert tpr to pdb, without selecting just water
		proc = pexpect.spawnu('%s trjconv -f %s -s %s -b 0 -e 0 -o %s' % (gmxExe,traj,tpr,outputFolder+'/system.pdb'))
		proc.send('0 0')
		proc.sendline()
		proc.wait()
		proc.kill(1)
		print('active')

		pdb = outputFolder+'/system_dry.pdb'
		pdbFull = outputFolder+'/system.pdb'
		copyfile(tpr,outputFolder+'/system.tpr')
		tpr = outputFolder+'/system.tpr'

	try:
		system = parsePDB(pdb)
	except:
		logger.exception('Could not load the PDB file. Aborting now.')
		return

	systemProtein = system.select('protein or nucleic')
	writePDB(outputFolder+'/system_dry.pdb',systemProtein)
	systemCA = system.select('name CA')
	numResidues = len(np.unique(systemProtein.getResindices()))

	for resindex in np.unique(systemProtein.getResindices()):
		residue = systemProtein.select('resindex %i' % resindex)
		index = np.unique(residue.getResnames())
		if len(index) > 1:
			logger.exception('There are residues with the same residue index in your PDB file. This is not allowed. Aborting now...')
			return

	paramFile = paramFile
	if paramFile and not type(paramFile) == str:
		paramFile = paramFile[0]

	if paramFile:
		paramFile = paramFile.split(' ')
		paramFile = [os.path.abspath(paramFile) for paramFile in paramFile]

	if not type(sourceSel) == str:
		if len(sourceSel) > 1:
			sourceSel = ' '.join(sourceSel)
		else:
			sourceSel = sourceSel[0]

	if targetSel and not type(targetSel) == str:
		if len(targetSel) > 1:
			targetSel = ' '.join(targetSel)
		else:
			targetSel = targetSel[0]

	if len(frameRange) > 1:
		frameRange = np.asarray(frameRange)
	else:
		frameRange = frameRange[0]

	if targetSel == False and pairCalc == True:
		logger.error("You must also specify a --targetsel for --paircalc to be executed,\
		 or vice versa. Aborting now.",exc_info=True)
		return

	if pairFilterCutoff < 4:
		logger.exception('Filtering distance cutoff value can not be smaller than 4. Aborting now.')

	# Load psf with prody and get some useful numbers.
	### COMMENTING THIS OUT DUE TO INCOMPATIBILITY PROBLEMS WITH python3.6
	### WE DON'T NEED TO USE PSF IN PYTHON ANYWAY
	# try:
	# 	system = parsePSF(psf)
	# except:
	# 	logger.exception('Could not load the PSF file provided. Please check your input PDB file.\
	# 		 Aborting now.')
	# 	return

	# Saving traj argument here to a string (cause we change it later on)
	trajPath = traj

	# Check whether the system has enough memory to multiple processing of the DCD
	trajStats = os.stat(trajPath)
	size = trajStats.st_size

	memory = psutil.virtual_memory()

	if not size*numCores > memory.available*1.1:
		logger.info('System has enough memory to handle the computation... Proceeding...')
	else:
		logger.exception('System does not have enough memory to handle the computation. \
			Please either decrease the number of processors (numCores) or reduce the size of input DCD trajectory. \
			Aborting now.')
		return

	# Check the input trajectory and convert to DCD if necessary.
	if not trajPath.endswith('.dcd') and (trajPath.endswith('.xtc') or trajPath.endswith('.trr')):
		# Convert XTC/TRR trajectories to DCD for ProDy compatible analysis...
		logger.info('Detected GMX trajectory... Converting to DCD to proceed further...')
		try:
			if traj.endswith('.xtc'):
				traj = mdtraj.load_xtc(trajPath,top=outputFolder+'/system.pdb',stride=skip)
				traj.save_trr(outputFolder+'/traj.trr')
				trajPath = outputFolder+'/traj.trr'
			elif traj.endswith('.trr'):
				traj = mdtraj.load_trr(trajPath,top=outputFolder+'/system.pdb',stride=skip)

			dataType = 'GMX' # Specify a data type to use later on!
		except:
			logger.exception('Could not load the trajectory file provided. Please check your trajectory.')
			return

		traj.save_dcd(outputFolder+'/traj.dcd')
		# Load back this DCD and continue with it (for code compatibility with ProDy)
		traj = parseDCD(outputFolder+'/traj.dcd')
		logger.info('DCD file conversion success.')

	else:
		try:
			traj = Trajectory(trajPath)
			traj.link(system)
			dataType = 'NAMD'
		except:
			logger.exception('Could not load the DCD file provided. Please check your input DCD file.')
			return

	logger.info('Deleting waters from the trajectory...')
	traj.setAtoms(system.select('protein'))
	writeDCD(outputFolder+'/traj_dry.dcd',traj,step=skip if dataType=='NAMD' else 1)
	logger.info('Deleting waters from the trajectory... Done.')

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
	traj.setAtoms(systemCA)
	coordSets = traj.getCoordsets()

	# Start a contact matrix (Kirchhoff matrix)
	kh = np.zeros((numResidues,numResidues))

	# Accumulate contact matrix as the sim progresses
	calculatedPercentage = 0
	monitor = 0

	for i in range(0,len(coordSets),pairFilterSkip):
		coordSet = coordSets[i]
		gnm = GNM('GNM')
		gnm.buildKirchhoff(coordSet,cutoff=pairFilterCutoff)
		kh = kh + gnm.getKirchhoff()
		monitor = monitor + pairFilterSkip
		calculatedPercentage = (float(monitor)/float(len(coordSets)))*100
		if calculatedPercentage > 100: calculatedPercentage = 100
		logger.info('Filtered pairs percentage: %s' % str(calculatedPercentage))

	# Get whether contacts are below cutoff for the specified percentage of simulation
	pairsFilteredFlag = np.abs(kh)/(len(traj)/float(pairFilterSkip)) > pairFilterCutoff*0.01

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

	logger.info('Number of interaction pairs selected after filtering step: %i' % len(pairsFiltered))

	# Start energy calculation in chunks
	pairsFilteredChunks = np.array_split(np.asarray(pairsFiltered),numCores)

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
			os._exit(0)
		signal.signal(signal.SIGINT, sig_int)
	    
	# Start a pool of processors
	if dataType == 'NAMD':

		pool = multiprocessing.Pool(numCores,worker_init)

		pool.map(calcEnergiesSingleCoreNAMD,
			zip(pairsFilteredChunks,itertools.repeat(top),
				itertools.repeat(outputFolder+'/system_dry.pdb'),
				itertools.repeat(trajPath),
				itertools.repeat(skip),
				itertools.repeat(frameRange),
				itertools.repeat(pairFilterCutoff),
				itertools.repeat(environment),
				itertools.repeat(soluteDielectric),
				itertools.repeat(solventDielectric),
				itertools.repeat(outputFolder),
				itertools.repeat(namd2exe),
				itertools.repeat(paramFile),
				itertools.repeat(logFile)))
		
		#pool.join()
		# Parse the specified outFolder after energy calculation is done.
		outFolderFileList = os.listdir(outputFolder)

		energiesFilePaths = list()
		for fileName in outFolderFileList:
			if fileName.endswith('energies.log'):
				energiesFilePaths.append(outputFolder+'/'+fileName)

		energiesFilePathsChunks = np.array_split(list(energiesFilePaths),numCores)

		parsedEnergiesResults = pool.starmap(parseEnergiesSingleCoreNAMD,
			zip(energiesFilePathsChunks,itertools.repeat(outputFolder+'/system_dry.pdb'),
				itertools.repeat(logFile)))

		parsedEnergies = dict()
		for parsedEnergiesResult in parsedEnergiesResults:
			parsedEnergies.update(parsedEnergiesResult)


		pool.close()
		pool.join()

	elif dataType == 'GMX':
		calcEnergiesGMX(pairsFiltered=pairsFiltered,topFilePath=top,pdbFilePath=outputFolder+'/system.pdb',tprFilePath=tpr,
			trajFilePath=trajPath,skip=skip,frameRange=frameRange,pairFilterCutoff=pairFilterCutoff,outputFolder=outputFolder,
			gmxExe=gmxExe,logFile=logFile,numCores=numCores)

		parsedEnergies = parseEnergiesGMX(gmxExe=gmxExe,pdb=outputFolder+'/system.pdb',pairsFiltered=pairsFiltered,outputFolder=outputFolder)
	
	# while not parsedEnergiesResults.ready():
	# 	print("num left: {}".format(parsedEnergiesResults._number_left))
	# 	time.sleep(1)

	# parsedEnergiesResults = parsedEnergiesResults.get()
	logger.info('Collecting results...')

	# Prepare a pandas data table from parsed energies, write it to new files depending on type of energy
	df_total = pandas.DataFrame()
	df_elec = pandas.DataFrame()
	df_vdw = pandas.DataFrame()
	for key,value in list(parsedEnergies.items()):
		df_total[key] = value['Total']
		df_elec[key] = value['Elec']
		df_vdw[key] = value['VdW']

	logger.info('Saving results to '+outputFolder+'/energies_intEnTotal.csv')
	df_total.to_csv(outputFolder+'/energies_intEnTotal.csv')
	logger.info('Saving results to '+outputFolder+'/energies_intEnElec.csv')
	df_elec.to_csv(outputFolder+'/energies_intEnElec.csv')
	logger.info('Saving results to '+outputFolder+'/energies_intEnVdW.csv')
	df_vdw.to_csv(outputFolder+'/energies_intEnVdW.csv')

	logger.info('Saving results to '+outputFolder+'.pickle')
	# If saving to a pickle is requested:
	if toPickle:
		file = open(outputFolder+'.pickle','wb')
		pickle.dump(parsedEnergies,file)
		file.close()

	logger.info('Getting mean interaction energies...')
	# Save average interaction energies as well!
	if toPickle:
		getResIntEnMean(outputFolder+'.pickle',pdb,prefix=outputFolder+'/energies')

	if resIntCorr:
		logger.info('Beginning residue interaction energy correlation calculation...')
		getResIntCorr.getResIntCorr(inFile=outputFolder+'/'+'energies_intEnTotal.csv',
			pdb=pdb,meanIntEnCutoff=resIntCorrAverageIntEnCutoff,
			outPrefix=outputFolder+'/energies',logFile=logFile)

	logger.info('Cleaning up...')
	# Delete all namd-generated energies file from output folder.
	for item in glob.glob(outputFolder+'/*_energies.log'):
		os.remove(item)

	for item in glob.glob(outputFolder+'/*temp*'):
		os.remove(item)

	for item in glob.glob(outputFolder+'/interact*'):
		os.remove(item)

	for item in glob.glob(outputFolder+'/*.trr'):
		os.remove(item)

	if os.path.exists(outputFolder+'/traj.dcd'):
		os.remove(outputFolder+'/traj.dcd')

	logger.info('FINAL: Computation sucessfully completed. Thank you for using gRINN.')
	
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

	parser.add_argument('--pdb',default=' ',type=str,nargs=1,help='Name of the corresponding pdb file of the \
		DCD trajectory')

	parser.add_argument('--tpr',default=' ',type=str,nargs=1,help='Name of the corresponding TPR file of the \
		XTC/TRR trajectory')

	parser.add_argument('--top',default=' ',type=str,nargs=1,help='Name of the corresponding psf file of the \
		DCD trajectory or top file of the XTC/TRR trajectory')

	parser.add_argument('--traj',default=' ',type=str,nargs=1,help='Name of the trajectory file')

	parser.add_argument('--numcores',default=[1],type=int,nargs=1,
		help='Number of CPU cores to be employed for energy calculation. If not specified, it \
		defaults to 1 (no parallel computation will be done). If specified, e.g. NUMCORES=n, \
		then the computational workloading will be distributed among n cores.')

	parser.add_argument('--environment',default=['vacuum'],choices=[['vacuum','implicit-solvent']],type=str,nargs=1,
		help='Environment representation used during interaction energy calculation.')

	parser.add_argument('--solutedielectric',default=[1],type=int,nargs=1,
		help='Solute dielectric constant')

	parser.add_argument('--solventdielectric',default=[78.5],type=int,nargs=1,
		help='Solvent dielectric constant')

	parser.add_argument('--sourcesel',default=['all'],nargs='+',help='A ProDy atom selection \
	 string which determines the first group of selected residues. ')

	parser.add_argument('--targetsel',default=False,type=str,nargs='+',help='A ProDy atom selection string \
		which determines the second group of selected residues.')

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
	
	parser.add_argument('--gmxexe',default=['gmx'],type=str,nargs=1,help='Path to the GMX executable')

	parser.add_argument('--parameterfile',default=[False],type=str,nargs='+',help='Path to the parameter file.')

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

	top = os.path.abspath(args.top[0])
	pdb = os.path.abspath(args.pdb[0])
	tpr = os.path.abspath(args.tpr[0])
	traj = os.path.abspath(args.traj[0])
	
	numCores = args.numcores[0]
	frameRange = args.framerange
	skip = args.skip[0]

	environment = args.environment[0]

	solventDielectric = args.solventdielectric[0]
	soluteDielectric = args.solutedielectric[0]

	pairCalc = args.paircalc
	pairFilterCutoff = args.pairfiltercutoff[0]
	pairFilterBasis = args.pairfilterbasis[0]
	pairFilterPercentage = args.pairfilterpercentage[0]
	pairFilterSkip = args.pairfilterskip[0]

	outputFolder = os.path.abspath(args.outfolder[0])

	namd2exe = os.path.abspath(args.namd2exe[0])

	gmxExe = os.path.abspath(args.gmxexe[0])

	logFile = os.path.abspath(args.logfile[0])

	frameRange = args.framerange

	sourceSel = args.sourcesel
	targetSel = args.targetsel

	paramFile = args.parameterfile

	resIntCorr = args.resintcorr 
	resIntCorrAverageIntEnCutoff = args.resintcorraverageintencutoff[0]

	getResIntEn(top=top,pdb=pdb,tpr=tpr,traj=traj,numCores=numCores,
		sourceSel=sourceSel,targetSel=targetSel,environment=environment,
		soluteDielectric=soluteDielectric,solventDielectric=solventDielectric,
		pairCalc=pairCalc,pairFilterCutoff=pairFilterCutoff,pairFilterBasis=pairFilterBasis,
		pairFilterPercentage=pairFilterPercentage,pairFilterSkip=pairFilterSkip,
		skip=skip,frameRange=frameRange,resIntCorr=resIntCorr,
		resIntCorrAverageIntEnCutoff=resIntCorrAverageIntEnCutoff,outputFolder=outputFolder,
		namd2exe=namd2exe,gmxExe=gmxExe,paramFile=paramFile,toPickle=args.topickle,logFile=logFile)