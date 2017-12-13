#!/usr/bin/env python
from prody import *
import numpy as np
import mdtraj, multiprocessing, pexpect, sys, itertools, argparse, os, pyprind, subprocess, \
re, pickle, types, logging, datetime, psutil, signal, time, pandas, glob, platform, \
traceback
from shutil import copyfile, rmtree
from common import *
import corr

def getResIntEnMean(intEnPickle,pdb,frameRange=False,prefix=''):

	# Load interaction energy pickle file
	intEnFile = open(intEnPickle,'rb')
	intEn = pickle.load(intEnFile)
	numFrames = len(intEn[list(intEn.keys())[0]]['Total'])

	if not frameRange:
		frameRange = [0,numFrames]

	# Get number of residues
	system = parsePDB(pdb)
	system_dry = system.select('protein or nucleic')
	system_dry = system_dry.select('not resname SOL')
	numResidues = len(np.unique(system_dry.getResindices()))

	# Start interaction energy variables
	intEnDict = dict()
	intEnDict['Elec'] = np.zeros((numResidues,numResidues))
	intEnDict['Frame'] = np.zeros((numResidues,numResidues))
	intEnDict['Total'] = np.zeros((numResidues,numResidues))
	intEnDict['VdW'] = np.zeros((numResidues,numResidues))

	progbar = pyprind.ProgBar(numResidues*numResidues)

	for i in range(numResidues):
		i_chainResnameResnum = getChainResnameResnum(system_dry,i)
		for j in range(numResidues):
			j_chainResnameResnum = getChainResnameResnum(system_dry,j)
			keyString = i_chainResnameResnum+'-'+j_chainResnameResnum
			if keyString in intEn:
				intEnDict['Elec'][i,j] = np.mean(intEn[keyString]['Elec'][frameRange[0]:frameRange[1]])
				intEnDict['Elec'][j,i] = np.mean(intEn[keyString]['Elec'][frameRange[0]:frameRange[1]])
				intEnDict['Total'][i,j] = np.mean(intEn[keyString]['Total'][frameRange[0]:frameRange[1]])
				intEnDict['Total'][j,i] = np.mean(intEn[keyString]['Total'][frameRange[0]:frameRange[1]])
				intEnDict['VdW'][i,j] = np.mean(intEn[keyString]['VdW'][frameRange[0]:frameRange[1]])
				intEnDict['VdW'][j,i] = np.mean(intEn[keyString]['VdW'][frameRange[0]:frameRange[1]])

			else:
				intEnDict['Elec'][i,j] = 0
				intEnDict['Elec'][j,i] = 0
				intEnDict['Total'][i,j] = 0
				intEnDict['Total'][j,i] = 0
				intEnDict['VdW'][i,j] = 0
				intEnDict['VdW'][j,i] = 0

			progbar.update()

	# Save to text
	np.savetxt('%s_intEnMeanTotal.dat' % prefix,intEnDict['Total'])
	np.savetxt('%s_intEnMeanVdW.dat' % prefix,intEnDict['VdW'])
	np.savetxt('%s_intEnMeanElec.dat' % prefix,intEnDict['Elec'])

	# Save in column format as well (only Totals for now)
	f = open('%s_intEnMeanTotal' % prefix+'List.dat','w')
	for i in range(0,len(intEnDict['Total'])):
		for j in range(0,len(intEnDict['Total'][i])):
			value = intEnDict['Total'][i,j]
			if value: # i.e. it it's not equal to zero
				f.write('%s\t%s\t%s\n' % (getChainResnameResnum(system_dry,i),getChainResnameResnum(system_dry,j),str(value)))

	f.close()
	
	return intEnDict

def prepareFilesNAMD(params):
	# Load the PDB and PSF files, get rid of non-protein sections.
	params.logger.info('Parsing PDB file...')
	pdb = parsePDB(params.pdb)
	pdbProtein = pdb.select('protein')
	writePDB(os.path.join(params.outFolder,'system_dry.pdb'),pdbProtein)

	params.logger.info('Parsing PSF file...')
	# The problem with PSF files is, there is no good package for manipulating them
	# in python. One option is to use ProDy parsePSF, but it does not read dihedrals etc.
	# which are required by NAMD.
	# I implemented a method in common module for deleting waters/ions from PSF while keeping
	# dihedrals etc. in the output file using vmd-python from Robin Metz.
	makeDryPSF(psf=params.top,pdb=params.pdb,outFolder=params.outFolder)

	params.logger.info('Reading DCD file...')
	# Load the DCD file, get rid of non-protein sections.
	traj = Trajectory(params.traj)
	traj.link(pdb)
	traj.setAtoms(pdbProtein)
	writeDCD(os.path.join(params.outFolder,'traj_dry.dcd'),traj,step=params.stride)

def prepareFilesGMX(params):
	pass

def calcEnergiesSingleCoreNAMD(args):
	# Input arguments
	pairsFiltered = args[0]
	params = args[1]
	psfFilePath = os.path.join(params.outFolder,'system_dry.psf')
	pdbFilePath = os.path.join(params.outFolder,'system_dry.pdb')
	dcdFilePath = os.path.join(params.outFolder,'traj_dry.dcd')
	skip = 1 # We implemented this stride (skip) in the DCD file already.
	pairFilterCutoff = params.pairFilterCutoff
	environment = 'vacuum'
	soluteDielectric = params.dielectric
	solventDielectric = 80

	outputFolder = os.path.abspath(params.outFolder)
	namd2exe = params.exe
	# paramFile is a list by default, so we should map to get abspath
	paramFile = params.parameterFile
	logFile = os.path.abspath(params.logFile)

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started an energy calculation thread.')

	# Defining a method to calculate energies in chunks (to show the progress on the screen).
	def calcEnergiesSingleChunk(pairsFiltered,psfFilePath,pdbFilePath,dcdFilePath,skip,
		pairFilterCutoff,environment,soluteDielectric,solventDielectric,outputFolder,namd2exe,paramFile,
		logger):
		for pair in pairsFiltered:
			# Write PDB files for pairInteractionGroup specification
			system = parsePDB(pdbFilePath)
			sel1 = system.select(str('resindex %i' % int(pair[0])))
			sel2 = system.select(str('resindex %i' % int(pair[1])))
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
				stdout=open(os.path.join(outputFolder,'%i_%i_energies.log' % (pair[0],pair[1])),'w'))
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
		try:
			calcEnergiesSingleChunk(pairsFilteredChunk,psfFilePath,pdbFilePath,dcdFilePath,skip,
				pairFilterCutoff,environment,soluteDielectric,solventDielectric,outputFolder,namd2exe,paramFile,logger)
		except (KeyboardInterrupt,SystemExit):
			logger.exception('Keyboard interrupt detected. Aborting now.')
			sys.exit(0)

		progBar.update()
		percent = percent + 10
		logger.info('Completed calculation percentage: %s' % percent)

	logger.info('Completed a pairwise energy calculation thread.')

def calcEnergiesNAMD(params):
	# Start energy calculation in chunks
	params.logger.info('Splitting the pairs into chunks...')
	params.pairsFilteredChunks = np.array_split(np.asarray(params.pairsFiltered),params.numCores)

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

	global pool
	pool = multiprocessing.Pool(params.numCores,worker_init)
 
	# Catching CTRL+C SIGINT signals.
	def sigint_handler(signum, frame):
		params.logger.exception('Keyboard interrupt detected. Aborting now.')
		global pool
		pool.terminate()
		#time.sleep(5)
		sys.exit(0)

	signal.signal(signal.SIGINT, sigint_handler)

	# Use map_aysnc on the previously created multiprocessing pool to spawn multiple singe core
	# energy calculation threads.
	# get(9999999) below is necessary to let the map respond without blocking the spawned threads.
	# This is a python bug in 2.7
	params.logger.info('Starting threads for interaction energy calculation...')
	# Strip logger away from params temporarily to be able to map.
	logger = params.logger
	params.logger = None
	results = pool.map_async(calcEnergiesSingleCoreNAMD,
			zip(params.pairsFilteredChunks,itertools.repeat(params))).get(9999999)
	params.logger = logger

	# Parse the specified outFolder after energy calculation is done.
	outFolderFileList = os.listdir(params.outFolder)

	energiesFilePaths = list()
	for fileName in outFolderFileList:
		if fileName.endswith('energies.log'):
			energiesFilePaths.append(os.path.join(params.outFolder,fileName))

	energiesFilePathsChunks = np.array_split(list(energiesFilePaths),
		params.numCores)

	parsedEnergiesResults = pool.map_async(parseEnergiesSingleCoreNAMD,
		zip(energiesFilePathsChunks,itertools.repeat(os.path.join(
			params.outFolder,'system_dry.pdb')),
			itertools.repeat(params.logFile))).get(9999999)

	parsedEnergies = dict()
	for parsedEnergiesResult in parsedEnergiesResults:
		parsedEnergies.update(parsedEnergiesResult)

	pool.close()
	pool.join()

	params.parsedEnergies = parsedEnergies
	return params

def calcEnergiesGMX(pairsFiltered,topFilePath,pdbFilePath,tprFilePath,trajFilePath,skip,frameRange,
	pairFilterCutoff,soluteDielectric,outputFolder,gmxExe,logFile,numCores):

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)

	logger.info('Started an energy calculation thread.')

	# Prevent backup making while calculating energies.
	os.environ["GMX_MAXBACKUP"] = "-1"

	# Make an index and MDP file with the pairs filtered.
	#gmxExe = 'gmx'
	mdpFiles,pairsFilteredChunks = makeNDXMDPforGMX(gmxExe=gmxExe,pdb=pdbFilePath,
		tpr=tprFilePath,soluteDielectric=soluteDielectric,pairsFiltered=pairsFiltered,outFolder=outputFolder,
		logger=logger)

	# Call gromacs pre-processor (grompp) and make a new TPR file for each pair and calculate energies for each pair.
	i = 0
	edrFiles = list()
	for i in range(0,len(mdpFiles)):
		mdpFile = mdpFiles[i]
		tprFile = mdpFile.rstrip('.mdp')+'.tpr'
		edrFile = mdpFile.rstrip('.mdp')+'.edr'

		args = [gmxExe,'grompp','-f',mdpFile,'-n',
			os.path.join(outputFolder,'interact.ndx'),'-p',topFilePath,'-c',tprFilePath,'-o',tprFile,'-maxwarn','20']
		proc = subprocess.Popen(args)
		proc.wait()

		proc = subprocess.Popen([gmxExe,'mdrun','-rerun',trajFilePath,'-s',tprFile,
			'-e',edrFile,'-nt',str(numCores)])
		proc.wait()

		edrFiles.append(edrFile)

		logger.info('Completed calculation percentage: '+str((i+1)/float(len(mdpFiles))*100))

	return edrFiles, pairsFilteredChunks

def filterPairs(params):
	
	system = parsePDB(os.path.join(params.outFolder,'system_dry.pdb'))
	traj = parseDCD(os.path.join(params.outFolder,'traj_dry.dcd'))

	try:
		sourceCA = system.select(str(params.sel1)+' and name CA')
	except:
		params.logger.exception('Could not select Selection 1 residue group. Aborting now.')
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
	
	# Get target selection residues
	try:
		targetCA = system.select(str(params.sel2+' and name CA'))
	except:
		params.logger.exception('Could not select Selection 2 residue group. Aborting now.')
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
	pairChunks = np.array_split(list(pairSet),params.numCores)

	params.logger.info('Starting the filtering step...')

	# Continue with filtering operation
	traj.setAtoms(allResiduesCA)
	coordSets = traj.getCoordsets()

	# Start a contact matrix (Kirchhoff matrix)
	kh = np.zeros((numResidues,numResidues))

	# Accumulate contact matrix as the sim progresses
	calculatedPercentage = 0
	monitor = 0

	for i in range(0,len(coordSets),1):
		coordSet = coordSets[i]
		gnm = GNM('GNM')
		gnm.buildKirchhoff(coordSet,cutoff=params.pairFilterCutoff)
		kh = kh + gnm.getKirchhoff()
		monitor = monitor + 1
		calculatedPercentage = (float(monitor)/float(len(coordSets)))*100
		if calculatedPercentage > 100: calculatedPercentage = 100
		params.logger.info('Filtered pairs percentage: %s' % str(calculatedPercentage))

	# Get whether contacts are below cutoff for the specified percentage of simulation
	pairsFilteredFlag = np.abs(kh)/(len(traj)/float(1)) > params.pairFilterCutoff*0.01

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
		params.logger.exception('Filtering step did not yield any pairs. '
			'Either your cutoff value is too small or the percentage criteria is too high.')
		return

	params.logger.info('Number of interaction pairs selected after filtering step: %i' % len(pairsFiltered))

	params.pairsFiltered = pairsFiltered
	return params

def collectResults(params):
	params.logger.info('Collecting results...')

	# Prepare a pandas data table from parsed energies, write it to new files depending on type of energy
	df_total = pandas.DataFrame()
	df_elec = pandas.DataFrame()
	df_vdw = pandas.DataFrame()
	for key,value in list(params.parsedEnergies.items()):
		df_total[key] = value['Total']
		df_elec[key] = value['Elec']
		df_vdw[key] = value['VdW']

	params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies_intEnTotal.csv'))
	df_total.to_csv(params.outFolder+'/energies_intEnTotal.csv')
	params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies_intEnElec.csv'))
	df_elec.to_csv(params.outFolder+'/energies_intEnElec.csv')
	params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies_intEnVdW.csv'))
	df_vdw.to_csv(params.outFolder+'/energies_intEnVdW.csv')

	params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies.pickle'))
	file = open(os.path.join(params.outFolder,'energies.pickle'),'wb')
	pickle.dump(params.parsedEnergies,file)
	file.close()

	params.logger.info('Getting mean interaction energies...')
	# Save average interaction energies as well!
	getResIntEnMean(os.path.join(params.outFolder,'energies.pickle'),
		os.path.join(params.outFolder,'system_dry.pdb'),
		prefix=os.path.join(params.outFolder,'energies'))

	return params
	if resIntCorr:
		logger.info('Beginning residue interaction energy correlation calculation...')
		getResIntCorr.getResIntCorr(inFile=os.path.join(
			outputFolder,'energies_intEnTotal.csv'),
			pdb=pdb,meanIntEnCutoff=resIntCorrAverageIntEnCutoff,
			outPrefix=os.path.join(outputFolder,'energies'),logger=logger)

def cleanUp(params):
	params.logger.info('Cleaning up...')
	# Delete all namd-generated energies file from output folder.
	for item in glob.glob(os.path.join(params.outFolder,'*_energies.log')):
		os.remove(item)

	for item in glob.glob(os.path.join(params.outFolder,'*temp*')):
		os.remove(item)

	for item in glob.glob(os.path.join(params.outFolder,'interact*')):
		os.remove(item)

	for item in glob.glob(os.path.join(params.outFolder,'*.trr')):
		os.remove(item)

	if os.path.exists(os.path.join(params.outFolder,'traj.dcd')):
		os.remove(os.path.join(params.outFolder,'traj.dcd'))

def calcNAMD(params):
	# Prepare input files for NAMD energy calculation.
	prepareFilesNAMD(params)

	# Filter pairs.
	params = filterPairs(params)

	# Calculate interaction energies.
	params = calcEnergiesNAMD(params)

	# Collect results.
	params = collectResults(params)

	# Clean up
	cleanUp(params)

def calcGMX(params):
	pass

# Method to convert TPR to PDB files.
def tpr2pdb(params,tpr,pdb,gmxGroup):
	# Convert tpr to pdb, selecting just Protein.
	# Apparently directly spawning gmx in the following does not work as expect in OSX
	# Prepending bash -c to the command line prior to gmx.
	proc = pexpect.spawnu('bash -c "%s trjconv -f %s -s %s -b 0 -e 0 -o %s"' % 
		(params.exe,params.traj,tpr,pdb))
	try:
		proc.expect(u'Select a group:.*')
		#proc.logfile = sys.stdout
	except pexpect.EOF:
		print('error')
		return False, proc.before

	proc.send(gmxGroup)
	proc.sendline()
	#proc.wait() # proc.wait() does not work on MacOSX for some reason...
	while not os.path.exists(pdb):
		time.sleep(1) # using time.sleep(X) instead, sleeping for X seconds to let the bg process complete work
		
	# Check whether the file is still being written to...
	while has_handle(pdb):
		time.sleep(1)

	proc.kill(1)
	return True, "Success'"

# Method to check args and get params if they are valid
def getParams(args):

	# Make a new parameters object.
	params = parameters()

	params.numCores = args.numcores[0]
	frameRange = args.framerange

	if len(frameRange) > 1:
		params.frameRange = np.asarray(frameRange)
	elif len(frameRange) == 1:
		if not frameRange[0]:
			params.frameRange = False
	else:
		message = 'Invalid frame range. Aborting now.'
		return params, False, message

	params.stride = args.stride[0]

	params.dielectric = args.dielectric[0]

	params.pairFilterCutoff = args.pairfiltercutoff[0]

	if params.pairFilterCutoff < 4:
		message = 'Filtering distance cutoff value can not be smaller than 4. Aborting now.'
		return params, False, message

	params.pairFilterPercentage = args.pairfilterpercentage[0]

	if not type(args.sel1) == str:
		if len(args.sel1) > 1:
			params.sel1 = ' '.join(args.sel1)
		else:
			params.sel1 = args.sel1[0]

	if not type(args.sel2) == str:
		if len(args.sel2) > 1:
			params.sel2 = ' '.join(args.sel2)
		else:
			params.sel2= args.sel2[0]

	# Check whether the output folder exists. If it exists, abort.
	outFolder = os.path.abspath(args.outfolder[0])
	currentFolder = os.getcwd()
	if outFolder != currentFolder:
		if os.path.exists(outFolder):
			print("The output folder exists. Please delete this folder or "
				" specify a folder path that does not exist. Aborting now.")
			sys.exit(0)
		else:
			params.outFolder = outFolder
			params.logFile = os.path.join(os.path.abspath(outFolder),'grinn.log')

	# Check input simulation data.
	if not args.top[0]:
		message = "You must specify a valid topology file (PSF or TOP). Aborting now."
		return params, False, message
	else:
		params.top = os.path.abspath(args.top[0])
		if params.top.lower().endswith('.psf'):
			try:
				topology = parsePSF(params.top)
			except:
				message = "Could not load your PSF file. Aborting now."
				return params, False, message

	if args.pdb[0] and args.tpr[0]:
		message = "You can't specify a PDB and a TPR file at the same time. Please specify either "
		"a PDB for NAMD data or a TPR for GROMACS data. Aborting now."
		return params, False, message

	if args.pdb[0]:
		try:
			system = parsePDB(os.path.abspath(args.pdb[0]))
			systemProtein = system.select(str('protein or nucleic'))
			params.pdb = os.path.abspath(args.pdb[0])
			params.dataType = 'namd'
		except:
			message = "Could not load your PDB file. Aborting now."
			return params, False, message

		try:
			sysSel1 = system.select(params.sel1)
			sysSel2 = system.select(params.sel2)
		except:
			message = 'Could not select sel1 or sel2 in the PDB file. Aborting now.'
			return params, False, message

		numResidues = len(np.unique(systemProtein.getResindices()))
		for resindex in np.unique(systemProtein.getResindices()):
			residue = systemProtein.select(str('resindex %i' % resindex))
			index = np.unique(residue.getResnames())
			if len(index) > 1:
				message = 'There are multiple residues with the same residue index in your PDB file. '
				' This is not allowed. Aborting now...'
				return params, False, message

	elif args.tpr[0]:
		# Unfortunately I don't know of a good way to check valid GMX tpr data.
		if not args.tpr[0].lower().endswith('.tpr'):
			message = "The TPR file must have extension .tpr. Aborting now."
		else:
			params.tpr = os.path.abspath(args.tpr[0])
			params.dataType = 'gmx'

	else:
		message = "Please specify either a PDB for NAMD data or a TPR for GROMACS data. "
		"Aborting now."
		return params, False, message

	if not args.traj[0]:
		message = "You have not specified a trajectory file!"
		return params, False, message
	else:
		params.traj = os.path.abspath(args.traj[0])

	# Check whether given exe is actually an exe!
	# If not, abort.
	if not args.exe[0]:
		message = "You have not specified a NAMD2 or GMX executable!"
		return params, False, message
	if os.path.exists(os.path.join(os.getcwd(),args.exe[0])):
		params.exe = os.path.abspath(args.exe[0])
	else:
		params.exe = args.exe[0]

	isExe = which(params.exe)
	if not isExe:
		message = "NAMD2/GMX exe you specified does not appear to be a valid executable. "
		"Aborting now."
		return params, False, message

	# Check extension combinations.
	_,trajExt = os.path.splitext(params.traj)
	if params.dataType == 'namd':
		_,topExt = os.path.splitext(params.top)
		_,pdbExt = os.path.splitext(params.pdb)
		exts = [topExt.lower(),pdbExt.lower(),trajExt.lower()]
		if exts != ['.psf','.pdb','.dcd']:
			message = 'Invalid PSF/PDB/DCD file extensions. Aborting now.'
			return params, False, message

		try:
			trajectory = parseDCD(params.traj)
		except:
			message = 'Could not load the DCD file provided. Aborting now.'
			return params, False, message

		# Check whether the system has enough memory for multiple processing of the DCD
		trajStats = os.stat(params.traj)
		size = trajStats.st_size

		memory = psutil.virtual_memory()
		if size*params.numCores > memory.available*1.1:
			message = 'System does not have enough memory to handle the computation. '
			'Please either decrease the number of processors (numCores) or reduce '
			'the size of input DCD trajectory. Aborting now.'
			return params, False, message

		# Check whether a parameter file is supplied.
		parameterFile = args.parameterfile
		for paramFile in parameterFile:
			if not paramFile:
				message = 'You must supply a parameter file for NAMD. Aborting now.'
				return params, False, message

		params.parameterFile = [os.path.abspath(paramFile) for paramFile in parameterFile]
		#print(params.parameterFile)
		#return params, False, "what the hell?"
		
	elif params.dataType == 'gmx':

		if platform.system() == 'Windows':
			message = 'GROMACS data on Windows is not supported. Aborting now.'
			return params, False, message

		_,tprExt = os.path.splitext(params.tpr)
		_,topExt = os.path.splitext(params.top)
		exts = [topExt.lower(),tprExt.lower(),trajExt.lower()]
		if exts != ['.top','.tpr','.trr'] and exts != ['.top','.tpr','.xtc']:
			message = 'Invalid TOP/TPR/XTC/TRR file extensions. Aborting now.'
			return params, False, message

		# Check whether a PDB can be extracted from the TPR.
		isPDB,messageOut = tpr2pdb(params,params.tpr,'dummy.pdb','System')
		if not isPDB:
			message = 'Could not extract a structure from input TPR.'
			message = message + ' Executable produced the following :\n' 
			message = message + messageOut
			return params, False, message
		else:
			try:
				system = parsePDB('dummy.pdb')
				systemProtein = system.select(str('protein or nucleic'))
				os.remove('dummy.pdb')
			except:
				os.remove('dummy.pdb')
				message = 'Could not load the extracted PDB file from TPR. '
				'Aborting now.'
				return params, False, message

			try:
				sysSel1 = system.select(params.sel1)
				sysSel2 = system.select(params.sel2)
			except:
				message = 'Could not select sel1 or sel2 in the PDB file extracted from the '
				'TPR. Aborting now.'
				return params, False, message

	return params, True, "Success"

# Main method starting the work
def getResIntEn(args):
	
	# Check whether input arguments are valid and get parameters!
	global params
	params, isArgsValid, message = getParams(args)

	# Create the output folder now so that we can start logging.
	# Creating this file right now is important because the calcGUI 
	# will monitor this file as well.
	os.makedirs(params.outFolder)
	f = open(params.logFile,'w')
	f.close()

	# Start logging.
	loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
	logging.basicConfig(format=loggingFormat,datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,
		filename=params.logFile)
	params.logger = logging.getLogger(__name__)
	
	# Also print messages to the terminal
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(logging.Formatter(loggingFormat))
	params.logger.addHandler(console)

	params.logger.info('Checking whether input arguments are valid...')

	# Check whether the arguments are valid. If not, remove the output folder and abort.
	if not isArgsValid:
		params.logger.exception(message)
		# Check whether the script was called from a terminal.
		if sys.stdin.isatty():
			rmtree(params.outFolder)
		return

	params.logger.info('Argument check completed. Proceeding...')

	params.logger.info('Started calculation.')

	# Proceed with the appropriate method depending on the input data type.
	if params.dataType == 'namd':
		calcNAMD(params)
	elif params.dataType == 'gmx':
		pass
		#calcGMX(params)
	
	params.logger.info('FINAL: Computation sucessfully completed. Thank you for using gRINN.')
	return

	###### WARNING !!! ################
	# BELOW THIS POINT EVIL LIVES!!! ###
	###### THE CODE IS LEFTOVER CODE ###
	
	if tpr:
		#gmxExe = 'gmx' # TEMPORARY

		logger.info('Detected TPR file, converting to PDB...')

		
		# Convert tpr to pdb, full system.
		proc = pexpect.spawnu('bash -c "%s trjconv -f %s -s %s -b 0 -e 0 -o %s"' % (gmxExe,traj,tpr,os.path.join(
			outputFolder,'system.pdb')))
		proc.expect(u'Select a group:.*')
		proc.logfile = sys.stdout
		proc.send('0 0')
		proc.sendline()

		# Check whether file has been created. If not, wait.
		while not os.path.exists(os.path.join(
			outputFolder,'system.pdb')):
			time.sleep(1)

		# Check whether the file is still being written to...
		while has_handle(os.path.join(
			outputFolder,'system.pdb')):
			time.sleep(1)

		proc.kill(1)

		logger.info('Detected TPR file, converting to PDB... Done.')
		pdb = os.path.join(outputFolder,'system.pdb')
		copyfile(tpr,os.path.join(outputFolder,'system.tpr'))
		tpr = os.path.join(outputFolder,'system.tpr')

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


	# Check the input trajectory and convert to DCD if necessary.
	if not trajPath.endswith('.dcd') and (trajPath.endswith('.xtc') or trajPath.endswith('.trr')):
		# Convert XTC/TRR trajectories to DCD for ProDy compatible analysis...
		logger.info('Detected GMX trajectory... Converting to DCD...')
		try:
			if traj.endswith('.xtc'):
				traj = mdtraj.load_xtc(trajPath,top=os.path.join(outputFolder,'system.pdb'),stride=skip)
				traj.save_trr(outputFolder+'/traj.trr')
				trajPath = outputFolder+'/traj.trr'
			elif traj.endswith('.trr'):
				traj = mdtraj.load_trr(trajPath,top=os.path.join(outputFolder,'system.pdb'),stride=skip)

			dataType = 'GMX' # Specify a data type to use later on!
		except:
			logger.exception('Could not load the trajectory file provided. Please check your trajectory.')
			return

		traj.save_dcd(os.path.join(outputFolder,'traj.dcd'))
		# Load back this DCD and continue with it (for code compatibility with ProDy)
		traj = Trajectory(os.path.join(str(outputFolder),'traj.dcd'))
		traj.link(system)
		logger.info('Detected GMX trajectory... Converting to DCD... Done.')
		#results.get(9999999)# gotta use this get at the end, because of a known Python2.7 bug.
		#results.wait()
		
		#pool.join()

	elif dataType == 'GMX':
		edrFiles,pairsFilteredChunks = calcEnergiesGMX(pairsFiltered=pairsFiltered,topFilePath=top,
			pdbFilePath=os.path.join(outputFolder,'system.pdb'),tprFilePath=tpr,trajFilePath=trajPath,skip=skip,
			frameRange=frameRange,pairFilterCutoff=pairFilterCutoff,soluteDielectric=soluteDielectric,outputFolder=outputFolder,
			gmxExe=gmxExe,logFile=logFile,numCores=numCores)

		parsedEnergies = parseEnergiesGMX(gmxExe=gmxExe,pdb=os.path.join(
			outputFolder,'system.pdb'),pairsFilteredChunks=pairsFilteredChunks,outputFolder=outputFolder,
			edrFiles=edrFiles,logger=logger)
	
	# while not parsedEnergiesResults.ready():
	# 	print("num left: {}".format(parsedEnergiesResults._number_left))
	# 	time.sleep(1)

	# parsedEnergiesResults = parsedEnergiesResults.get()

if __name__ == '__main__':
	print('Please do not call this script directly. Use python grinn.py -calc <arguments> '
		'instead.')
	sys.exit(0)


