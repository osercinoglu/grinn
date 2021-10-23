#!/usr/bin/env /home/onur/anaconda3/bin/python
# -*- coding: utf-8 -*-
from prody import *
from prody import LOGGER
import logging
# Directly modifying logging level for ProDy to prevent printing of noisy debug/warning
# level messages on the terminal.
LOGGER._logger.setLevel(logging.FATAL)
import numpy as np
import mdtraj, pexpect, sys, itertools, argparse, os, pyprind, subprocess, \
re, pickle, types, datetime, psutil, signal, time, pandas, glob, platform, \
traceback, click, copy, math
from shutil import copyfile, rmtree
from itertools import islice
from concurrent import futures
from scipy.sparse import lil_matrix
from common import *
import corr
from memory_profiler import profile

def convert_arg_line_to_args(arg_line):
    # To override the same method of the ArgumentParser (to read options from a file)
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def getResIntEnMean(intEnPicklePaths,pdb,sel1,sel2,frameRange=False,prefix=''):

    # Load interaction energy pickle files
    # First, load the first dictionary (there must be at least one).
    intEn = pickle.load(open(intEnPicklePaths[0],'rb'))
    numFrames = len(intEn[list(intEn.keys())[0]]['Total'])

    # Then update its content with the rest.
    for fpath in intEnPicklePaths[1:]:
        intEnFile = open(fpath,'rb')
        intEn2update = pickle.load(intEnFile)
        intEn.update(intEn2update)
        
    if not frameRange:
        frameRange = [0,numFrames]

    # Get number of residues
    with suppress_stdout():
        system = parsePDB(pdb)

    system_source = system.select(sel1)
    system_target = system.select(sel2)

    resindices_source = np.unique(system_source.getResindices())
    numSource = len(resindices_source)

    resindices_target = np.unique(system_target.getResindices())
    numTarget = len(resindices_target)

    # Start interaction energy variables
    intEnDict = dict()

    # Each dict key here will hold a matrix for different IE types.
    # It's a good idea to save residue indices in first row and column here!
    # So that in results gui we can detect interacting pairs easily.
    # For this, matrix sizes are adjusted (+1) below.
    template_mat = np.zeros((numSource+1,numTarget+1))
    template_mat[:,0] = np.hstack([0,resindices_source])
    template_mat[0,:] = np.hstack([0,resindices_target])
    intEnDict['Elec'] = template_mat
    intEnDict['Frame'] = template_mat
    intEnDict['Total'] = template_mat
    intEnDict['VdW'] = template_mat

    progbar = pyprind.ProgBar(numSource*numTarget)

    filteredButNoInt = list() # Accumulate interactions that were included in calculation but resulted in zero interaction energy.
    for i in range(numSource):
        for j in range(numTarget):
            keyString = str(resindices_source[i])+'-'+str(resindices_target[j])
            if keyString in intEn:
                intEnDict['Elec'][i+1,j+1] = np.mean(intEn[keyString]['Elec'][frameRange[0]:frameRange[1]])
                #intEnDict['Elec'][j+1,i+1] = np.mean(intEn[keyString]['Elec'][frameRange[0]:frameRange[1]])
                totalMeanEn = np.mean(intEn[keyString]['Total'][frameRange[0]:frameRange[1]])
                intEnDict['Total'][i+1,j+1] = totalMeanEn
                #intEnDict['Total'][j+1,i+1] = totalMeanEn
                intEnDict['VdW'][i+1,j+1] = np.mean(intEn[keyString]['VdW'][frameRange[0]:frameRange[1]])
                #intEnDict['VdW'][j+1,i+1] = np.mean(intEn[keyString]['VdW'][frameRange[0]:frameRange[1]])

                if not totalMeanEn:
                    filteredButNoInt.append(keyString)

            else:
                intEnDict['Elec'][i+1,j+1] = int(0)
                #intEnDict['Elec'][j+1,i+1] = int(0)
                intEnDict['Total'][i+1,j+1] = int(0)
                #intEnDict['Total'][j+1,i+1] = int(0)
                intEnDict['VdW'][i+1,j+1] = int(0)
                #intEnDict['VdW'][j+1,i+1] = int(0)

            progbar.update()

    # Save to text
    np.savetxt('%s_intEnMeanTotal.dat' % prefix,intEnDict['Total'])
    np.savetxt('%s_intEnMeanVdW.dat' % prefix,intEnDict['VdW'])
    np.savetxt('%s_intEnMeanElec.dat' % prefix,intEnDict['Elec'])

    # Save in column format as well (only Totals for now)
    f = open('%s_intEnMeanTotal' % prefix+'List.dat','w')

    # Below we start the ranges from 1 because the first row and column were spared for residue indices.
    for i in range(1,len(intEnDict['Total'])):
        for j in range(1,len(intEnDict['Total'][i])):
            value = intEnDict['Total'][i,j]
            if value: # i.e. if it's not equal to zero (included in filtering step or included but was zero)
                source_index = intEnDict['Total'][i][0]
                target_index = intEnDict['Total'][0][j]
                f.write('%s\t%s\t%s\n' % (str(source_index),str(target_index),str(value)))

    f.close()
    
    return intEnDict, filteredButNoInt

def isStructureDry(pdb,psf):
    # Load the PDB and PSF files
    with suppress_stdout():
        pdb = parsePDB(pdb)
    pdbDry = pdb.select('protein or nucleic or hetero or lipid and not water and not resname SOL and not ion')
    pdbNonDry = pdb.select('water or resname SOL or ion')

    with suppress_stdout():
        psf = parsePSF(psf)
    psfNonDry = psf.select('water or resname SOL or ion')

    if not pdbNonDry == None and not psfNonDry == None:
        return False
    else:
        return True

def prepareFilesNAMD(params):
    # Detect whether there are non-protein components in the system.
    params.logger.info('Checking whether the structure has non-protein atoms...')

    #### TEMP MODIFICATION: CANCELING DRY STRUCTURE CHECK NOW.
    # dryStructure = isStructureDry(params.pdb,params.top)

    # if not dryStructure:
    #   params.logger.info('Non-protein atoms detected in structure...')
    #   params.logger.info('There are non-protein elements in your input files. Please '
    #       'consider generating PSF/PDB/DCD files containing only the protein in your '
    #       'structure.') # Maybe add here a link where the user can check his/her options.
    #### TEMP MODIFICATION

    if sys.stdin.isatty():
        if not click.confirm('Do you want to continue?', default=True):
            errorSuicide(params,'User requested abort. Aborting now.',removeOutput=False)

    # Proceeding.
    # Just copy psf and pdb files and the trajectory with stride to output folder.
    copyfile(params.pdb,os.path.join(params.outFolder,'system.pdb'))
    copyfile(params.top,os.path.join(params.outFolder,'system.psf'))
    # Load the DCD file, get rid of non-protein sections.
    with suppress_stdout():
        traj = Trajectory(params.traj)
        pdb = parsePDB(params.pdb)
        traj.link(pdb)
        traj.setAtoms(pdb)
        writeDCD(os.path.join(params.outFolder,'traj.dcd'),
            traj,step=params.stride)
        # Load it back, superpose, save again.
        traj = parseDCD(os.path.join(params.outFolder,'traj.dcd'))
        traj.setAtoms(pdb)
        traj.superpose()
        writeDCD(os.path.join(params.outFolder,'traj.dcd'),traj)

    # Check whether system has enough memory to handle the computation...
    proceed, message = isMemoryEnough(params,os.path.join(params.outFolder,'traj.dcd'))
    if not proceed:
        errorSuicide(params,message,removeOutput=False)

def prepareFilesGMX(params):
    params.logger.info('Converting TPR to PDB...')

    # Convert tpr to pdb, full system.
    isPDB,messageOut = tpr2pdb(params,params.tpr,
        os.path.join(params.outFolder,'system.pdb'))

    # Check whether file has been created. If not, wait.
    while not os.path.exists(os.path.join(
        params.outFolder,'system.pdb')):
        time.sleep(1)

    # Check whether the file is still being written to...
    while has_handle(os.path.join(
        params.outFolder,'system.pdb')):
        time.sleep(1)

    params.pdb = os.path.join(params.outFolder,'system.pdb')

    params.logger.info('Converting TPR to PDB/GRO... Done.')
    pdb = os.path.join(params.outFolder,'system.pdb')
    copyfile(params.tpr,os.path.join(params.outFolder,'system.tpr'))
    tpr = os.path.join(params.outFolder,'system.tpr')

    # Make dry PDB out of the resulting PDB.
    with suppress_stdout():
        pdb = parsePDB(params.pdb)
        pdbDry = pdb.select('protein or nucleic or lipid or hetero and not water and not resname SOL and not ion')
        writePDB(os.path.join(params.outFolder,'system_dry.pdb'),pdbDry)

    # Convert XTC/TRR trajectories to DCD for ProDy compatible analysis...
    params.logger.info('Converting XTC/TRR to DCD...')
    try:
        if params.traj.endswith('.xtc'):
            traj = mdtraj.load_xtc(params.traj,
                top=os.path.join(params.outFolder,'system.gro'),
                stride=params.stride)
        elif params.traj.endswith('.trr'):
            traj = mdtraj.load_trr(params.traj,
                top=os.path.join(params.outFolder,'system.gro'),
                stride=params.stride)

        traj.save_trr(os.path.join(params.outFolder,'traj.trr'))
        params.traj = os.path.join(params.outFolder,'traj.trr')

        dataType = 'GMX' # Specify a data type to use later on!
    except:
        params.logger.exception('Could not load the trajectory file provided. Please check your trajectory.')
        return

    with suppress_stdout():
        traj.save_dcd(os.path.join(params.outFolder,'traj.dcd'))
        # Load back this DCD and continue with it (for code compatibility with ProDy)
        traj = Trajectory(os.path.join(str(params.outFolder),'traj.dcd'))
        traj.link(pdb)
        traj.setAtoms(pdbDry)

        # Write
        writeDCD(os.path.join(params.outFolder,'traj_dry.dcd'),traj)

        # Load it back and superpose, then write back.
        traj = parseDCD(os.path.join(params.outFolder,'traj_dry.dcd'))
        traj.setAtoms(pdbDry)
        traj.superpose()
        writeDCD(os.path.join(params.outFolder,'traj_dry.dcd'),traj)
        
    os.remove(os.path.join(params.outFolder,'traj.dcd'))
    params.logger.info('Converting to XTC/TRR to DCD... Done.')

    return params

def calcEnergiesSingleCoreNAMD(args):
    # Input arguments
    pairsFilteredSingleCore = args[0]
    params = args[1]
    psfFilePath = os.path.join(params.outFolder,'system.psf')
    pdbFilePath = os.path.join(params.outFolder,'system.pdb')
    dcdFilePath = os.path.join(params.outFolder,'traj.dcd')
    skip = 1 # We implemented this stride (skip) in the DCD file already.
    pairFilterCutoff = params.pairFilterCutoff
    cutoff = params.cutoff
    switchdist = params.switchdist
    environment = 'vacuum'
    soluteDielectric = params.dielectric
    solventDielectric = 80

    outputFolder = os.path.abspath(params.outFolder)
    namd2exe = params.exe
    # paramFile is a list by default, so we should map to get abspath
    paramFile = params.parameterFile
    logger = params.logger
    #logFile = os.path.abspath(params.logFile)

    #loggingFormat = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    #logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    #    datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
    #logger = logging.getLogger(__name__)

    # Also print messages to the terminal
    #console = logging.StreamHandler()
    #console.setLevel(logging.INFO)
    #console.setFormatter(logging.Formatter(loggingFormat))
    #logger.addHandler(console)

    logger.info('Started an energy calculation thread.')

    # Defining a method to calculate energies in chunks (to show the progress on the screen).
    def calcEnergiesSingleChunk(pairsFilteredSingleChunk,psfFilePath,pdbFilePath,dcdFilePath,skip,
        pairFilterCutoff,cutoff,switchdist,environment,soluteDielectric,solventDielectric,
        outputFolder,namd2exe,paramFile,logger):

        for pair in pairsFilteredSingleChunk:
            # Write PDB files for pairInteractionGroup specification
            with suppress_stdout():
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
                # Delete system, sel1, and sel2 to release pressure on RAM.
                del system
                del sel1
                del sel2

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
            f.write('exclude scaled1-4\n')
            f.write('outputname %i_%i-temp\n' % (pair[0],pair[1]))
            f.write('temperature 0\n')
            f.write('COMmotion yes\n')
            f.write('cutoff %f\n' % cutoff)
            
            if environment == 'implicit-solvent':
                f.write('GBIS on\n')
                f.write('solventDielectric %d\n' % solventDielectric)
                f.write('dielectric %d\n' % soluteDielectric)
                f.write('alphaCutoff %d\n' % (float(cutoff)-3)) # Setting GB radius to cutoff for now. We might want to change this behaviour later.
                f.write('SASA on\n')
            elif environment == 'vacuum':
                f.write('dielectric %d\n' % soluteDielectric)
            else:
                f.write('#environment is %s\n' % str(environment))

            if switchdist:
                f.write('switching on\n')
                f.write('switchdist %f\n' % switchdist)
            else:
                f.write('switching off\n')
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
            #f.write('\tcoorfile skip\n') # Don't need it once you apply stride to tray_dry.dcd in outputfolder
            f.write('}\n')
            f.write('coorfile close')
            f.close()

            # Run namd2 to compute the energies
            try:
                pid_namd2 = subprocess.Popen([namd2exe,'+p%i' % params.namd2NumCores,namdConf],
                    stdout=open(
                        os.path.join(outputFolder,'%i_%i_energies.log' % 
                            (pair[0],pair[1])),'w'),stderr=subprocess.PIPE)
                _,error = pid_namd2.communicate()
            except KeyboardInterrupt:
                print('Keyboard interrupt detected.')
                sys.exit(0)

            if error:
                print(error)
                logger.exception('Error while calling NAMD executable:\n'+str(error))
                error = error.decode().split('\n')
                fatalErrorLine = None

                for i in range(0,len(error)):
                    if 'FATAL ERROR:' in error[i]:
                        fatalErrorLine = error[i]
                        continue

                if fatalErrorLine:
                    return fatalErrorLine

            pid_namd2.wait()

            logger.info('Energies saved to %i_%i_energies.log' % (pair[0],pair[1]))
            if not os.path.exists(os.path.join(params.outFolder,'%i_%i_energies.log' % (pair[0],pair[1]))):
                return "gRINN was supposed to generate %i_%i_energies.log but apparently it failed." % (pair[0],pair[1])

            # Clean up already at this point to avoid disk-space devouring behaviour for very large systems/long trajs.
            temp_fns = '%s_%s-temp.*' % (pair[0],pair[1])
            for item in glob.glob(os.path.join(params.outFolder,temp_fns)):
                os.remove(item)

        return None

            #subprocess.call('rm %s' % namdConf,shell=True)
            #subprocess.call('rm %s' % pairIntPDB,shell=True)
            #subprocess.call('rm %i_%i-temp*' % (pair[0],pair[1]),shell=True)
            #raise SystemExit(0)
        # Parse the log file and extract necessary energy values

        # Done.

    # Split the pairsFiltered into chunks to print the progress on the screen.
    if len(pairsFilteredSingleCore) >= 100:
        numChunks = 100
    elif len(pairsFilteredSingleCore) < 10:
        numChunks = 1
    else:
        numChunks = 10

    pairsFilteredChunksSingleCore = np.array_split(pairsFilteredSingleCore,numChunks)

    progBar = pyprind.ProgBar(numChunks)

    # Perform the calculations in chunks
    percent = 0

    for pairsFilteredChunk in pairsFilteredChunksSingleCore:
        try:
            errorMessage = calcEnergiesSingleChunk(pairsFilteredChunk,psfFilePath,pdbFilePath,dcdFilePath,skip,
                pairFilterCutoff,cutoff,switchdist,environment,soluteDielectric,solventDielectric,
                outputFolder,namd2exe,paramFile,logger)
        except (SystemExit):
            #logger.exception('Fatal error while calling NAMD executable.')
            return 'SystemExit'

        if errorMessage:
            return errorMessage

        progBar.update()
        percent = percent + 100/float(numChunks)
        logger.info('Completed calculation percentage: %s' % percent)
        #print('Completed calculation percentage: %s' % percent)

        #################
        ### DEBUGGING: Printing list of local variables' sizes to find out what's devouring RAM here.
        # local_vars = list(locals().items())
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in local_vars),
        # 	key= lambda x: -x[1])[:10]:
        # 	print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        # global_vars = list(globals().items())
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in global_vars),
        # 	key= lambda x: -x[1])[:10]:
        # 	print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        ### DEBUGGING END ---------------------------------------------------------------------------
        #################

    logger.info('Completed a pairwise energy calculation thread.')
    # Necessary to proceed in parent method?
    return None

def parseEnergiesNAMD(params):
    
    # Parse the specified outFolder after energy calculation is done.
    outFolderFileList = os.listdir(params.outFolder)
    energiesFilePaths = list()
    
    for fileName in outFolderFileList:
        if fileName.endswith('energies.log'):
            energiesFilePaths.append(os.path.join(params.outFolder,fileName))

    energiesFilePathsChunks = np.array_split(list(energiesFilePaths),
        params.numCores)

    with futures.ProcessPoolExecutor(params.numCores) as pool:

        # Cancelling the following for map_async, it was intended for python 2.7
        #parsedEnergiesResults = pool.map_async(parseEnergiesSingleCoreNAMD,
        #   zip(energiesFilePathsChunks,itertools.repeat(os.path.join(
        #       params.outFolder,'system.pdb')),
        #       itertools.repeat(params.logFile))).get(9999999)
        
        # Instead, the following line.
        parsedEnergiesResults = pool.map(parseEnergiesSingleCoreNAMD,
            zip(energiesFilePathsChunks,itertools.repeat(os.path.join(
                params.outFolder,'system.pdb')),
                itertools.repeat(params.logFile)))
        parsedEnergiesResults = list(parsedEnergiesResults)

    parsedEnergies = dict()
    for parsedEnergiesResult in parsedEnergiesResults:
        parsedEnergies.update(parsedEnergiesResult)

    # Update parsedEnergies dict in the main params object.
    params.parsedEnergies.update(parsedEnergies)

    return params

def calcEnergiesNAMD(params):

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

    # Catching CTRL+C SIGINT signals.
    def sigint_handler(signum, frame):
        params.logger = logger
        global pool
        pool.terminate()
        pool.join()
        pool.close()
        print('signal: %s' % signum)
        parent = psutil.Process(os.getpid())
        children = parent.children()
        # Loop over children.
        for child in children:
            while child.children():
                # If child has children, note them
                for grandchild in child.children():
                    print("killing grandchild: %s" % grandchild.pid)
                    try:
                        grandchild.kill()
                    except:
                        pass

            if child.pid != os.getpid():
                print("killing child: %s" % child.pid)
                child.kill()

        #time.sleep(5)
        if sys.stdin.isatty():
            if not click.confirm('Would you like to delete the output folder?', default=True):
                errorSuicide(params,'Keyboard interrupt detected. Aborting now.',removeOutput=False)
            else:
                errorSuicide(params,'Keyboard interrupt detected. Aborting now.',removeOutput=True)
        else:
            errorSuicide(params,'GUI interrupt detected. Aborting now.',removeOutput=False)
        
        #print("killing parent: %s" % parent_id)
        #parent.kill()
        #print("suicide: %s" % os.getpid())
        #psutil.Process(os.getpid()).kill()
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    global pool

    params.logger.info('Starting threads for interaction energy calculation...')

    # Get possible pairs for which IE calculation has already been complete.
    if params.pairsProcessed.size > 0: # Then we have previously processed IE pairs.
        pairs2process = [pair for pair in params.pairsFiltered if pair not in params.pairsProcessed]
    else: # Then we're starting anew.
        pairs2process = params.pairsFiltered

    # Start energy calculation in chunks
    params.logger.info('Splitting the pairs into chunks...')
    arrPairsFiltered = np.asarray(pairs2process)

    ## Experimental: reduce params.numCores by half and call namd2 with +pNAMD2NUMCORES later on
    # This is to reduce RAM usage which sometimes chokes gRINN.
    origNumCores = params.numCores
    newNumCores = math.floor(params.numCores/params.namd2NumCores)
    params.numCores = newNumCores

    # Split pairs list into chunks adjusting number of pairs per core.
    numPairsFiltered = len(arrPairsFiltered)
    numPairsPerCore = math.floor(numPairsFiltered/params.numCores)

    if numPairsPerCore>= 5:
        numPairsPerCore = 5
    
    params.pairsFilteredChunks = np.array_split(arrPairsFiltered,
        int(len(arrPairsFiltered)/(numPairsPerCore*params.numCores)))

    # Run pool.map for each chunk over a for loop.
    progbar = pyprind.ProgBar(len(params.pairsFilteredChunks))

    #for chunk in [params.pairsFilteredChunks[0]]:
    for chunk in params.pairsFilteredChunks:
        params.logger.info('A chunk of pairs is being processed... Size of chunk: %i pairs.' % len(chunk))
        chunkSplitted = np.array_split(chunk,params.numCores)
        # Strip logger away from params temporarily to be able to map.
        logger = params.logger
        #params.logger = None

        # Reset numCores to the calculated value above (it is reset before params.pkl saving -below- to allow resuming later on).
        params.numCores = newNumCores

        # Start a concurrent.futures pool.
        with futures.ProcessPoolExecutor(params.numCores) as pool:
            results = pool.map(calcEnergiesSingleCoreNAMD,zip(chunkSplitted,itertools.repeat(params)))
            results = list(results)

        #params.logger = logger
    
        # If the pool return at least one 'SystemExit' string
        # Abort
        # see the calcEnergiesSingleCoreNAMD method)
        
        if 'SystemExit' in results:
            removeOutput = False if sys.stdin.isatty() else False
            errorSuicide(params,'Critical error while calling NAMD executable. \n\n'
                'Error could not be identified in detail. Please inspect your input data carefully.\n'
                'If the error persists, contact us. Aborting now.',
                removeOutput=removeOutput)
        elif results[0] is not None:
            if 'FATAL ERROR: ' in results[0]:
                removeOutput = False if sys.stdin.isatty() else False
                errorMessage = results[0] # Cause with multiple CPUs multiple outputs are possible.
                errorSuicide(params,'Fatal error from NAMD: '+
                    errorMessage.lstrip('FATAL ERROR:'),removeOutput=removeOutput)

        ## Collect results here, and update params.pkl in the output folder as well!
        params = parseEnergiesNAMD(params)

        # Update processed pairs at this point.
        params.pairsProcessed = np.vstack((params.pairsProcessed,chunk)) if params.pairsProcessed.size > 0 else chunk

        # Also save the params object so that it might be loaded later on after this run is interrupted to resume calculation.
        params.logger.info('Updating the params.pkl...')
        params.numCores = origNumCores
        with open(os.path.join(os.path.abspath(params.outFolder),'params.pkl'),'wb') as f:
            pickle.dump(params,f)
        params.logger.info('Updating the params.pkl... Done.')

        cleanUp(params)

        params.logger.info('A chunk of pairs processed.')
        progbar.update()

    # Reset params.numCores
    params.numCores = origNumCores

    return params

def calcEnergiesGMX(params):

    params.logger.info('Started an energy calculation thread.')

    # Prevent backup making while calculating energies.
    os.environ["GMX_MAXBACKUP"] = "-1"

    # Make an index and MDP file with the pairs filtered.
    #gmxExe = 'gmx'
    mdpFiles,pairsFilteredChunks = makeNDXMDPforGMX(gmxExe=params.exe,
        pdb=params.pdb,tpr=params.tpr,soluteDielectric=params.dielectric,
        pairsFiltered=params.pairsFiltered,outFolder=params.outFolder,
        logger=params.logger)

    # Call gromacs pre-processor (grompp) and make a new TPR file for each pair and calculate energies for each pair.
    i = 0
    edrFiles = list()
    for i in range(0,len(mdpFiles)):
        mdpFile = mdpFiles[i]
        tprFile = mdpFile.rstrip('.mdp')+'.tpr'
        edrFile = mdpFile.rstrip('.mdp')+'.edr'

        args = [params.exe,'grompp','-f',mdpFile,'-n',
            os.path.join(params.outFolder,'interact.ndx'),'-p',params.top,'-c',
            params.tpr,'-o',tprFile,'-maxwarn','20']
        proc = subprocess.Popen(args)
        proc.wait()

        # Catching CTRL+C SIGINT signals.
        def sigint_handler(signum, frame):
            proc.kill()
            if sys.stdin.isatty():
                if not click.confirm('Would you like to delete the output folder?', default=True):
                    errorSuicide(params,'Keyboard interrupt detected. Aborting now.',removeOutput=False)
                else:
                    errorSuicide(params,'Keyboard interrupt detected. Aborting now.',removeOutput=True)
            else:
                errorSuicide(params,'GUI interrupt detected. Aborting now.',removeOutput=False)

        signal.signal(signal.SIGINT,sigint_handler)

        proc = subprocess.Popen([params.exe,'mdrun','-rerun',params.traj,'-v','-s',tprFile,
            '-e',edrFile,'-nt',str(params.numCores)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)

        error = proc.communicate()[1]
        if error:
            error = error.decode().split('\n') # Splitting into lines to be able to process each line separately.
            fatalErrorLines = None

            # Collect fatal error and subsequent lines.
            for j in range(0,len(error)):
                if 'Fatal error' in error[j]:
                    fatalErrorLines = error[j:]
                    continue

            if fatalErrorLines:
                fatalError = '\n'.join(fatalErrorLines)
                message = 'Fatal error from gmx:\n\n' + fatalError
                errorSuicide(params,message)
            # else:
            #   error = '\n'.join(error)
            #   message = 'Error from gmx:\n\n' + error
            #   errorSuicide(params,message)
                
        proc.wait()

        edrFiles.append(edrFile)

        params.logger.info('Completed calculation percentage: '+str((i+1)/float(len(mdpFiles))*100))

    return edrFiles, pairsFilteredChunks

# A method for initial filtering using a single core.
def filterInitialPairsSingleCore(args):

    outFolder = args[0]
    pairs = args[1]
    initPairFilterCutoff = args[2]
    with suppress_stdout():
        system = parsePDB(os.path.join(outFolder,'system.pdb'))

    # Define a method for initial filtering of a single pair.
    def filterInitialPair(outFolder,pair):
        com1 = calcCenter(system.select("resindex %i" % pair[0]))
        com2 = calcCenter(system.select("resindex %i" % pair[1]))
        dist = calcDistance(com1,com2)
        if dist <= 30:
            return pair
        else:
            return None

    # Get a list of included pairs after initial filtering.
    filterList = list()
    progbar = pyprind.ProgBar(len(pairs))
    for pair in pairs:
        filtered = filterInitialPair(outFolder,pair)
        if filtered is not None:
            filterList.append(pair)
        progbar.update()

    return filterList

# A method for initial filtering.
def filterInitialPairs(params):
    
    with suppress_stdout():
        params.system = parsePDB(os.path.join(params.outFolder,'system.pdb'))

    try:
        params.source = params.system.select(str(params.sel1))
    except:
        params.logger.exception('Could not select Selection 1 residue group. Aborting now.')
        return

    #NEEDS CAREFUL TWEAKING ALL BELOW
    params.numResidues = params.system.numResidues()
    params.sourceResids = np.unique(params.source.getResindices())
    params.sourceResnums = [params.source.select('resindex %i' % i).getResnums()[0] for i in params.sourceResids]
    params.sourceSegnames = [params.source.select('resindex %i' % i).getSegnames()[0] for i in params.sourceResids]

    params.numSource = len(params.sourceResids)

    # By default, targetResids are all residues.
    params.targetResids = np.arange(params.numResidues)
    params.numTarget = len(params.targetResids)
    
    # Get target selection residues
    try:
        params.target = params.system.select(str(params.sel2))
        params.targetResids = np.unique(params.target.getResindices())
        params.numTarget = len(params.targetResids)
    except:
        params.logger.exception('Could not select Selection 2 residue group. Aborting now.')
        return

    # Generate all possible unique pairwise residue-residue combinations
    pairProduct = itertools.product(params.sourceResids,params.targetResids)
    pairSet = set()
    for x,y in pairProduct:
        if x != y:
            pairSet.add(frozenset((x,y)))

    params.logger.info('Starting filtering operations...')

    # Prepare a pairSet list.
    params.logger.info('Preparing a list of pairs...')
    pairSet = [list(pair) for pair in list(pairSet)]
    params.logger.info('Preparing a list of pairs... Done.')

    # Get a list of pairs within a certain distance from each other, 
    # based on the initial structure.
    params.initialFilter = list()

    # Initial filtering of pairs
    params.logger.info('Starting initial filtering step...')

    params.logger.info('Parallelizing initial filtering calculation...')

    # Split the pair set list into chunks according to number of cores
    pairChunks = np.array_split(list(pairSet),params.numCores)

    # Perform initial filtering on each of these chunks.
    params.logger.info('Performing initial filtering now... This may take a while...')

    # Start a concurrent futures pool, and perform initial filtering.
    with futures.ProcessPoolExecutor(params.numCores) as pool:
        params.initialFilter = pool.map(filterInitialPairsSingleCore,[[params.outFolder,pairChunks[i],params.initPairFilterCutoff] for i in range(0,params.numCores)])
        params.initialFilter = list(params.initialFilter)
        if len(params.initialFilter) > 1:
            params.initialFilter = np.vstack(params.initialFilter)

    params.initialFilter = list(params.initialFilter)
    params.initialFilter = [pair for pair in params.initialFilter if pair is not None]
    params.logger.info('Initial filtering... Done.')
    params.logger.info('Number of interaction pairs selected after initial filtering step: %i' %
        len(params.initialFilter))

    params.initialFilterDone = True

    with open(os.path.join(os.path.abspath(params.outFolder),'params.pkl'),'wb') as f:
        pickle.dump(params,f)

    return params

# A method for filtering using a single core.
def filterPairsSingleCore(args):
    
    params = args[0]
    trajIndex = args[1]
    numSource = args[2][0]
    numTarget = args[2][1]
    sourceResids = args[2][2]
    targetResids = args[2][3]
    initialFilter = args[3]

    with suppress_stdout():
        traj = parseDCD(os.path.join(params.outFolder,'traj_%i.dcd' % trajIndex))
        system = parsePDB(os.path.join(params.outFolder,'system.pdb'))

    coordSets = traj.getCoordsets()
    # Start a contact matrix based on center of masses
    contactMat = lil_matrix((numSource,numTarget))
    #contactMat = np.zeros((numSource,numTarget))

    # Accumulate contact matrix as the sim progresses
    calculatedPercentage = 0
    monitor = 0

    # Define a method for filtering of pairs based on pairFilterCutoff:
    def filterSinglePair(pair,pairFilterCutoff):
        pair1 = system.select('resindex %i' % pair[0])
        traj.setAtoms(pair1)
        pair1.setCoords(traj.getCoords())
        pair2 = system.select('resindex %i' % pair[1])
        traj.setAtoms(pair2)
        pair2.setCoords(traj.getCoords())
        com1 = calcCenter(pair1)
        com2 = calcCenter(pair2)
        dist = calcDistance(com1,com2)
        if dist <= pairFilterCutoff:
            if pair[0] in sourceResids and pair[1] in targetResids:
                source_index = list(sourceResids).index(pair[0])
                target_index = list(targetResids).index(pair[1])
            elif pair[1] in sourceResids and pair[0] in targetResids:
                source_index = list(sourceResids).index(pair[1])
                target_index = list(targetResids).index(pair[0])

            return [source_index,target_index]
        else:
            return None

    for i in range(0,len(coordSets),1):
        contactMatFrame = lil_matrix((numSource,numTarget))
        #contactMatFrame = np.zeros((numSource,numTarget))
        traj.setAtoms(system)
        traj.setCoords(traj.getCoordsets()[i])
        filteredPairIndices = list(map(filterSinglePair,initialFilter,
            [params.pairFilterCutoff]*len(initialFilter)))
        filteredPairIndices = [el for el in filteredPairIndices if el != None]
        for [source_index,target_index] in filteredPairIndices:
            contactMatFrame[source_index,target_index] = 1

        ### RED ALERT:
        ### The following may choke computer if the matrices are extremely large.
        ### For now we will take that the user makes a reasonable size selection for interaction energies.  
        contactMat = contactMat + contactMatFrame
        monitor = monitor + 1
        calculatedPercentage = (float(monitor)/float(len(coordSets)))*100
        if calculatedPercentage > 100: calculatedPercentage = 100
        #params.logger.info('Filtered pairs percentage: %s' % str(calculatedPercentage))

    del traj

    return contactMat

# A method for filtering of pairs.
def filterPairs(params):

    params.logger.info('Starting the filtering step...')

    # Split the trajectory into chunks according to number of cores.
    params.logger.info('Chunkifying trajectory...')
    traj = parseDCD(os.path.join(params.outFolder,'traj.dcd'))

    if len(traj) < params.numCores:
    	frameRanges = np.array_split(list(range(len(traj))),len(traj))
    else:
    	frameRanges = np.array_split(list(range(len(traj))),params.numCores)

    for i in range(0,len(frameRanges)):
        frameRange = frameRanges[i]
        if len(frameRange) == 1:
        	conf_i = traj[frameRange[0]]
        	traj_i = Ensemble()
        	traj_i.addCoordset(conf_i)
        elif len(frameRange) > 1:
        	traj_i = traj[frameRange[0]:frameRange[-1]]
        else:
        	continue

        writeDCD(os.path.join(params.outFolder,'traj_%i.dcd' % i),traj_i)
        del traj_i

    params.logger.info('Chunkifying trajectory... Done.')

    params.logger.info('Performing filtering now... This may take a while...')

    # Split initialFilter list into chunks, corresponding to 1000 pairs per numCores
    params.logger.info('Chunkifying initial filter...')
    chunkNum = 1 if len(params.initialFilter) < 500*params.numCores else len(params.initialFilter)/(500*params.numCores)
    initialFilterChunks = np.array_split(params.initialFilter,chunkNum)

    # Run pool.map for each chunk over a for loop.
    progbar = pyprind.ProgBar(len(initialFilterChunks))
    contactMaps = list()

    for i in range(0,len(initialFilterChunks)):
        chunk = initialFilterChunks[i]
        params.logger.info('Filtering a pair chunk...(%i out of %i)' % (i+1,len(initialFilterChunks)))

        # Start a concurrent.futures pool, and perform filtering.
        with futures.ProcessPoolExecutor(params.numCores) as pool:

            try:
                # For each traj portion split above
                ### POTENTIAL IMPROVEMENT BELOW: PASS ONLY PARAMS INSTEAD OF numSource etc. (assuming they've been moved under params.)
                contactMapsTrajChunk = pool.map(
                    filterPairsSingleCore,[[params,i,[params.numSource,params.numTarget,params.sourceResids,params.targetResids],
                    chunk] for i in range(0,params.numCores)])
                contactMapsTrajChunk = list(contactMapsTrajChunk)
            
                # In case multiple cores are selected, contactMapsTrajChun will have a size larger than
                # one. In this case, we need to sum contact map matrices.
                if len(contactMapsTrajChunk) > 1:
                    contactMapsTrajChunk = sum(contactMapsTrajChunk)

                params.logger.info('Filtering a pair chunk... Done.')
                contactMaps.append(contactMapsTrajChunk)

            finally:
                pool.shutdown()
        progbar.update()

    # In case multiple chunks are present, contactMaps will have a size large than one.
    # In this case, we need to sum contact map matrices.
    if len(contactMaps) > 1:
        contactMaps = sum(contactMaps)
    # else, if the contactMaps has a size of 1, the first member of this list is taken.
    elif len(contactMaps) == 1:
        contactMaps = contactMaps[0]

    # Get whether contacts are below cutoff for the specified percentage of simulation
    pairsInclusionFraction = np.abs(contactMaps)/(len(traj)/float(1))
    pairsFilteredFlag = pairsInclusionFraction > params.pairFilterPercentage*0.01

    ###################################################
    ### BELOW BLOCK SHOULD BE ACCELERATED (A LOT)! ####
    params.pairsFiltered = list()
    #concatSourceTargetResids = np.concatenate([sourceResids,targetResids])
    progbar = pyprind.ProgBar(len(params.sourceResids))
    params.logger.info('Collecting filtered pairs now...')

    for sourceResid in params.sourceResids:
        for targetResid in params.targetResids:
            if sourceResid == targetResid:
                continue
            source_index = list(params.sourceResids).index(sourceResid)
            target_index = list(params.targetResids).index(targetResid)

            if pairsFilteredFlag[source_index,target_index] > 0:
                params.pairsFiltered.append(sorted([sourceResid,targetResid]))

        progbar.update()

    params.logger.info('Collecting filtered pairs now... Done.')
    params.logger.info('Sorting filtered pairs now...')
    params.pairsFiltered = sorted(params.pairsFiltered)
    params.pairsFiltered = [list(x) for x in set(tuple(x) for x in params.pairsFiltered)]
    params.logger.info('Sorting filtered pairs now... Done.')
    ### END OF BLOCK TO BE ACCELERATED ###
    ######################################
    
    # file = open('pairsFiltered.txt','w')
    # for pair in pairsFiltered:
    #   file.write('%.2f-%i-%i\n' % (pairsInclusionFraction[pair[0],pair[1]],pair[0],pair[1]))
    # file.close()

    if not params.pairsFiltered:
        params.logger.exception('Filtering step did not yield any pairs. '
            'Either your cutoff value is too small or the percentage value is too high.')
        return

    params.logger.info('Number of interaction pairs selected after filtering step: %i' % len(params.pairsFiltered))

    # In some edge cases, the number of interactions pairs selected at this stage may really be so slow that it is lower than the numCores specified.
    # For these cases we'll just reduce the number of cores used, and make note of this in the log file.
    if len(params.pairsFiltered) < params.numCores:
        params.numCores = len(params.pairsFiltered)
        params.logger.info('The number of interaction pairs selected after filtering step is lower than the number of cores requested for calculation. '
        'Reducing number of cores to %i.' % len(params.pairsFiltered))

    params.filterDone = True

    with open(os.path.join(os.path.abspath(params.outFolder),'params.pkl'),'wb') as f:
        pickle.dump(params,f)

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
    df_total.to_csv(os.path.join(params.outFolder,'energies_intEnTotal.csv'))
    params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies_intEnElec.csv'))
    df_elec.to_csv(os.path.join(params.outFolder,'energies_intEnElec.csv'))
    params.logger.info('Saving results to '+os.path.join(params.outFolder,'energies_intEnVdW.csv'))
    df_vdw.to_csv(os.path.join(params.outFolder,'energies_intEnVdW.csv'))

    params.logger.info('Pickling results...')

    # Define a method for splitting dictionary into chunks.
    def chunks(data, SIZE=10000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k:data[k] for k in islice(it, SIZE)}

    # Split the dictionary into 10 chunks, this is for managing very large file sizes.
    # The dictionaries containing energies should be merged upon reading for analysis.
    enDicts = list()
    for enDict in chunks(params.parsedEnergies, 1000):
        enDicts.append(enDict)

    # Pickle the chunks.
    intEnPicklePaths = list()
    for i in range(0,len(enDicts)):
        fpath = os.path.join(params.outFolder,'energies_%i.pickle' % i)
        file = open(fpath,'wb')
        params.logger.info('Pickling to energies_%i.pickle...' % i)
        pickle.dump(enDicts[i],file)
        file.close()
        intEnPicklePaths.append(fpath)

    params.logger.info('Pickling results... Done.')

    params.logger.info('Getting mean interaction energies...')
    # Save average interaction energies as well!
    intEnDict, filteredButNoInt = getResIntEnMean(intEnPicklePaths,
        os.path.join(params.outFolder,'system.pdb'),params.sel1,params.sel2,
        prefix=os.path.join(params.outFolder,'energies'))

    # Report interactions with zero mean.
    for noint in filteredButNoInt:
        params.logger.info('The interaction %s was included in energy calculation but yielded '
            'zero kcal/mol mean interaction energy.' % noint)   

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

    # Cancelled trajectory deleting from the folder as this is needed in case of resume.
    #if os.path.exists(os.path.join(params.outFolder,'traj.dcd')):
    #    os.remove(os.path.join(params.outFolder,'traj.dcd'))

def errorSuicide(params,message,removeOutput=False):
    params.logger.exception(message)
    if removeOutput:
        rmtree(params.outFolder)
    #psutil.Process(os.getpid()).kill()
    # Exit normally after printing the error to the log file.
    os._exit(0)

def calcNAMD(params):
    
    if params.resume == True:
        if params.initialFilterDone == False:
            params = filterInitialPairs(params)

        if params.filterDone == False:
            params.logger.info('Resuming from the filtering step now...')
            params = filterPairs(params)

        if params.calcDone == True:
            # Give message to the user that the calculations seems to be complete, 
            # nothing else to do. Then, abort.
            params.logger.error('gRINN seems to have finished the calculation already. Resuming calculation not possible. Aborting now.')
            sys.exit(0)

    else:
        # Prepare input files for NAMD energy calculation.
        prepareFilesNAMD(params) 

        # Filter pairs.
        params = filterInitialPairs(params)
        params = filterPairs(params)
    
    pairsFiltered = copy.deepcopy(params.pairsFiltered)

    # Calculate interaction energies.
    params = calcEnergiesNAMD(params)
    if not pairsFiltered == params.pairsFiltered:
        params.logger.error('pairsFiltered do not match...')

    # Collect results.
    params = collectResults(params)

    # Clean up
    cleanUp(params)

    return params

def calcGMX(params):
    # Prepare input files for GMX energy calculation.
    params = prepareFilesGMX(params)

    # Filter pairs.
    params = filterPairs(params)

    # Calculate interaction energies.
    edrFiles,pairsFilteredChunks = calcEnergiesGMX(params)

    # Parse resulting energy EDR files.
    params.parsedEnergies = parseEnergiesGMX(gmxExe=params.exe,
        pdb=os.path.join(params.outFolder,'system.pdb'),
        pairsFilteredChunks=pairsFilteredChunks,
        outputFolder=params.outFolder,edrFiles=edrFiles,
        logger=params.logger)

    # Collect results
    params = collectResults(params)

    # Clean up
    cleanUp(params)

    return params

# Method to convert TPR to PDB files.
def tpr2pdb(params,tpr,pdb):
    # Convert tpr to pdb and gro, selecting just Protein.
    # Apparently directly spawning gmx in the following does not work as expect in OSX
    # Prepending bash -c to the command line prior to gmx.

    filenames = [pdb,pdb.rstrip('.pdb')+'.gro']
    for filename in filenames:
        proc = subprocess.Popen('bash -c "%s editconf -f %s -o %s"' % 
            (params.exe,tpr,filename),stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,shell=True)
        _,error = proc.communicate()
        #if error:
        #   message = repr(error)
        #   return False, message

        proc.wait()
        start_time = time.time()
        time_elapsed = 0 # Seconds
        while not os.path.exists(filename) and time_elapsed < 30:
            time.sleep(1) # using time.sleep(X) instead, sleeping for X seconds to let the bg process complete work
            time_elapsed = time.time() - start_time

        if not os.path.exists(filename):
            message = 'Could not extract PDB/GRO out of TPR file. Aborting now.'
            return False, message

        # Check whether the file is still being written to...
        while has_handle(filename):
            time.sleep(1)

    # Check whether there any chain ids not assigned a valid letter.
    noChid = False
    system = parsePDB(pdb)
    systemDry = system.select('protein or nucleic or lipid or hetero and not water and not resname SOL and not ion')
    chids = systemDry.getChids()
    for chid in chids:
        if not chid.strip():
            noChid = True
    
    if noChid:
        if params.logger: # Considering the case when this method is called during argument checking step..
            params.logger.info('At least one atom with no chain IDs present. Assigning the default chain ID P to such atoms right now...')
        system_noChids = system.select('chain " "')
        system_noChids.setChids(['P']*system_noChids.numAtoms())
        writePDB(pdb,system)            

    return True, "Success'"

# Method to write parameters to the log file.
def params2log(params):

    params.logger.info('Using the following input files and parameters:')
    params.logger.info('PDB: %s' % str(params.pdb))
    params.logger.info('TPR: %s' % str(params.tpr))
    params.logger.info('TOP: %s' % str(params.top))
    params.logger.info('Trajectory: %s' % str(params.traj))
    params.logger.info('Number of cores: %s' % str(params.numCores))
    params.logger.info('Solute dielectric (NAMD): %s' % str(params.dielectric))
    params.logger.info('Switch distance (NAMD): %s' % str(params.switchdist))
    params.logger.info('Selection 1: %s' % str(params.sel1))
    params.logger.info('Selection 2: %s' % str(params.sel2))
    params.logger.info('Filtering distance cutoff (Angstroms): %s' % str(params.pairFilterCutoff))
    params.logger.info('Filtering percentage (%%): %s' % str(params.pairFilterPercentage))
    params.logger.info('Non-bonded distance cutoff (Angstroms, NAMD): %s' % str(params.cutoff))
    params.logger.info('Trajectory stride: %s' % str(params.stride))
    params.logger.info('Executable: %s' % str(params.exe))
    params.logger.info('Parameter file(s) (NAMD) %s' % str(params.parameterFile))
    params.logger.info('Correlation: %s' % str(params.calcCorr))
    params.logger.info('Correlation cutoff: %s' % str(params.corrIntenCutoff))
    params.logger.info('Output folder: %s' % str(params.outFolder))

# Method to check args and get params if they are valid
def getParams(args):

    # Make a new parameters object.
    params = parameters()

    params.resume = args.resume

    # Check whether the output folder exists. 
    outFolder = os.path.abspath(args.outfolder[0])
    currentFolder = os.getcwd()
    if outFolder != currentFolder:

        # If outFolder exists and resume is not requested:
        if os.path.exists(outFolder) and params.resume==False:
            print("The output folder exists. Please delete this folder or "
                " specify a folder path that does not exist. Aborting now.")
            sys.exit(1)
        # If outFolder exists and resume is requested:
        elif os.path.exists(outFolder) and params.resume == True:
            if not os.path.exists(os.path.join(os.path.abspath(outFolder),'params.pkl')):
                print("params.pkl is not present in the output folder. This file is needed to resume calculation. "
                    "Aborting now. ")
                sys.exit(1)

            with open(os.path.join(os.path.abspath(outFolder),'params.pkl'),'rb') as f:
                params = pickle.load(f)
                params.resume = True # This may have been False in the saved params.pkl, if this is the first resume of calc.
                return params, True, 'Success' # Return params already.

        elif not os.access(os.path.abspath(
            os.path.dirname(outFolder)), os.W_OK):
            print("Can't write to the output folder path. Do you have write access?")
            return
        else:
            params.resume = False # If outfolder does not exist, resume can't be applied.
            params.outFolder = outFolder
            params.logFile = os.path.join(os.path.abspath(outFolder),'grinn.log')

    # Check whether any input file paths include space character.
    files = [args.pdb,args.tpr,args.top,args.traj]
    files = [filename[0] for filename in files if not filename[0] == None]
    files = [filename for filename in files if type(filename) == str]
    for filename in files:
        if ' ' in filename:
            message = "A file path (%s) includes a space character, which is not allowed. "\
            "Aborting now" % filename
            return params, False, message

    params.numCores = args.numcores[0]
    params.namd2NumCores = args.namd2numcores[0]
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

    params.initPairFilterCutoff = args.initpairfiltercutoff[0]

    if params.initPairFilterCutoff < 15:
        message = 'Initial filtering distance cutoff value can not be smaller than 15. Aborting now.'
        return params, False, message

    params.pairFilterCutoff = args.pairfiltercutoff[0]

    if params.pairFilterCutoff < 4:
        message = 'Filtering distance cutoff value can not be smaller than 4. Aborting now.'
        return params, False, message

    params.pairFilterPercentage = args.pairfilterpercentage[0]

    params.cutoff = args.cutoff[0]

    params.switchdist = args.switchdist[0]

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
            with suppress_stdout():
                system = parsePDB(os.path.abspath(args.pdb[0]))
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

        #### TEMP MODIFICATION: CANCELLING THIS NOW.
        # Check whether there any chain ids not assigned a valid letter.
        # chids = system.getChids()
        # for chid in chids:
        #   if not chid.strip():
        #       message = 'There is at least one residue with no chain ID assigned to it. This is not '\
        #       'allowed. Aborting now...'
        #       return params, False, message
        #### TEMP MODIFICATION: CANCELLING THIS NOW.

        numResidues = len(np.unique(system.getResindices()))
        for resindex in np.unique(system.getResindices()):
            residue = system.select(str('resindex %i' % resindex))
            index = np.unique(residue.getResnames())
            if len(index) > 1:
                message = 'There are multiple residues with the same residue index in your PDB file. '\
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
            trajectory = Trajectory(params.traj)
        except:
            message = 'Could not load the DCD file provided. Aborting now.'
            return params, False, message

        # Check whether stride is higher than the number of frames in trajectory:
        if params.stride > trajectory.numFrames():
            message = 'Stride value is higher than the number of frames in the trajectory. '\
            'Please use a lower stride value.'
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

        # Check whether switching is requested and if yes, whether it is smaller than 
        # or equal to the non-bonded cutoff.
        if params.switchdist:
            if params.switchdist > params.cutoff:
                message = 'Switch distance must be smaller than or equal to the non-bonded'\
                ' cutoff distance. Aborting now.'
                return params, False, message
        
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
        isPDB,messageOut = tpr2pdb(params,params.tpr,'dummy.pdb')
        if not isPDB:
            message = 'Could not extract a structure from input TPR.'
            message = message + ' Executable produced the following : ' 
            message = message + messageOut
            return params, False, message
        else:
            try:
                print('parsing dummy.pdb')
                system = parsePDB('dummy.pdb')
                print('selecting dry system')
                systemDry = system.select(str('protein or nucleic or lipid or hetero and not water and not resname SOL and not ion'))
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

    params.calcCorr = args.calccorr
    params.corrIntenCutoff = args.corrintencutoff

    return params, True, "Success"

# Main method starting the work
def getResIntEn(args):
    
    # Check whether input arguments are valid and get parameters!
    #global params // Might want to avoid declaring global here as params is passed around to subprocesses quite frequently.
    print('Checking input arguments...')
    params, isArgsValid, message = getParams(args)

    print('params.resume',params.resume)
    if params.resume == False:
        # Create the output folder now so that we can start logging.
        # Creating this file right now is important because the calcGUI 
        # will monitor this file as well.
        try:
            os.makedirs(params.outFolder)
            f = open(params.logFile,'w')
            f.close()
        except:
            print('Failed to create the output directory. Do you have write access?')
            sys.exit(0)

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

    # Check whether the arguments are valid. If not, remove the output folder and abort.
    if not isArgsValid:
        # Check whether the script was called from a terminal.
        if sys.stdin.isatty():
            errorSuicide(params,message,removeOutput=False)
            return
        else:
            errorSuicide(params,message,removeOutput=False)
            return

    params.logger.info('Argument check completed. Proceeding...')

    # Write parameters to the log file.
    params2log(params)

    params.logger.info('Started calculation.')

    # Proceed with the appropriate method depending on the input data type.
    if params.dataType == 'namd':
        params = calcNAMD(params)
    elif params.dataType == 'gmx':
        params = calcGMX(params)

    # Get correlations, if the user requested.
    if params.calcCorr:
        args.corrprefix = [os.path.join(params.outFolder,'energies')]
        args.corrinfile = [os.path.join(params.outFolder,'energies_intEnTotal.csv')]
        args.pdb = [params.pdb]
        corr.getResIntCorr(args,logFile=None,logger=params.logger)

    # Finalizing
    params.calcDone = True
    with open(os.path.join(os.path.abspath(params.outFolder),'params.pkl'),'wb') as f:
        pickle.dump(params,f)

    params.logger.info('FINAL: Computation sucessfully completed. Thank you for using gRINN.')
    return

if __name__ == '__main__':

    import argparse, common, sys, multiprocessing, os
    # Construct an argument parser.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='gRINN: get Residue Interaction eNergies and Networks. '
                                                 'gRINN calculates pairwise molecular-mechanics interaction energies '
                                                 'between amino acid residues in the context of molecular dynamics '
                                                 'simulations. gRINN also calculates equal-time correlations between '
                                                 'interaction energies and constructs protein energy networks. '
                                                 'gRINN offers separate graphical user interfaces for calculation of '
                                                 'energies and visualization of results.')

    # Overriding convert_arg_line_to_args in the input parser.
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('-calc', action='store_true', default=False,
                        help='Activates interaction energy calculation mode. ')

    parser.add_argument('-corr', action='store_true', default=False,
                        help='Activates interaction energy correlation calculation mode. ')

    parser.add_argument('-results', action='store_true', default=False,
                        help='Activates viewing of results mode. ')

    parser.add_argument('--pdb', default=[False], type=str, nargs=1,
                        help='Name of the corresponding PDB file of the DCD trajectory. '
                             'Applies only to NAMD-generated trajectories.')

    parser.add_argument('--tpr', default=[False], type=str, nargs=1,
                        help='Name of the corresponding TPR file of the XTC/TRR trajectory. '
                             'Applies only to GROMACS-generated trajectories.')

    parser.add_argument('--top', default=[False], type=str, nargs=1,
                        help='Name of the corresponding PSF file of the DCD trajectory or '
                             'TOP file of the XTC/TRR trajectory')

    parser.add_argument('--traj', default=[False], type=str, nargs=1,
                        help='Name of the trajectory file')

    parser.add_argument('--numcores', default=[multiprocessing.cpu_count()],
                        type=int, nargs=1,
                        help='Number of CPU cores to be employed by gRINN. '
                             'If not specified, it defaults to the number of cpu cores present '
                             'in your computer.')

    parser.add_argument('--namd2numcores', default=[1], type=int, nargs=1,
                        help='Number of CPU cores to be employed for interaction energy calculation '
                             'by NAMD2 executable in a single subprocess. If not specified, it defaults to 1, '
                             'and NUMCORES subprocesses are used.')

    parser.add_argument('--dielectric', default=[1], type=int, nargs=1,
                        help='Solute dielectric constant to be used in electrostatic interaction energy '
                             'computation. Applies only to NAMD-generated trajectories (DCD).')

    parser.add_argument('--switchdist', default=[False], type=float, nargs=1,
                        help='Switch distance (Angstroms) at which smoothing functions should be applied '
                             'on van der Waals and electrostatic forces. Applied only to NAMD-generated '
                             'trajectories (DCD). If not specified, switching is not applied, and a truncated '
                             'cutoff is applied.')

    parser.add_argument('--sel1', default=['all'], nargs='+',
                        help='A ProDy atom selection string which determines the first group of selected '
                             'residues. ')

    parser.add_argument('--sel2', default=['all'], type=str, nargs='+',
                        help='A ProDy atom selection string which determines the second group of selected '
                             'residues.')

    
    parser.add_argument('--initpairfiltercutoff', type=float, default=[30], nargs=1,
                        help='Cutoff distance (angstroms) for initial filtering of residues to be included in '
                        'the subsequent filtering step for interaction energy calculations. '
                        'If not specified, it defaults to 30 Angstroms. '
                        'Only those residues whose centers of mass are within the INITPAIRFILTERCUTOFF distance of each other '
                        'in the supplied PBD/TPR will be included in the subsequent filtering step.')

    parser.add_argument('--pairfiltercutoff', type=float, default=[15], nargs=1,
                        help='Cutoff distance (angstroms) for pairwise interaction energy calculations. '
                             'If not specified, it defaults to 15 Angstroms. '
                             'Only those residues whose centers of mass are within the PAIRFILTERCUTOFF distance of each other '
                             'for at least PAIRCUTOFFPERCENTAGE of the trajectory will be included '
                             'in energy calculations.')

    parser.add_argument('--pairfilterpercentage', type=float, default=[75], nargs=1,
                        help='When given, residues whose centers of masses	are within the PAIRFILTERCUTOFF distance from each '
                             'other for at least PAIRFILTERPERCENTAGE percent of the trajectory will be taken '
                             'into account in further evaluations. When not given, it defaults to 75%%)')

    parser.add_argument('--cutoff', type=float, default=[12], nargs=1,
                        help='Non-bonded interaction distance cutoff (Angstroms) for NAMD-type data.')

    parser.add_argument('--stride', default=[1], type=int, nargs=1,
                        help='If specified, a stride with value of STRIDE will be applied to the trajectory '
                             'during interaction energy calculation.')

    parser.add_argument('--framerange', type=int, default=[False], nargs='+',
                        help='If specified, then only FRAMERANGE\ section of the trajectory will be handled. '
                             'For example, if you specify --framerange 100 1000, then only frames between 100 '
                             ' and 1000 will be included in all calculations. Applies only to grinn -calc '
                             '<arguments> calls.')

    parser.add_argument('--exe', default=[None], type=str, nargs=1,
                        help='Path to the namd2/gmx executable. Defaults to namd2 or gmx, depending on '
                             'whether you specify NAMD2 or GMX type trajectory data (assumes namd2 or gmx is '
                             'in the executable search path.')

    parser.add_argument('--parameterfile', default=[False], type=str, nargs='+',
                        help='Path to the parameter file(s). Applies only to NAMD-generated data. '
                             'You can specify multiple parameters one after each other by placing a blank '
                             'space between them, e.g. --parameterfile file1.inp file2.inp. Applies only to '
                             'grinn -calc <arguments> calls.')

    parser.add_argument('--calccorr', action='store_true', default=False,
                        help='When specified, interaction energy correlation is also calculated following '
                             'interaction energy calculation in -calc mode. Equivalent to a grinn -corr call '
                             'after grinn -calc. Applies only to grinn -calc <arguments> calls.')

    parser.add_argument('--corrinfile', type=str, nargs=1, help='Path to the CSV file where interaction\
		energies are located in')

    parser.add_argument('--corrintencutoff', default=[1], type=float, nargs=1,
                        help='Mean (average) interaction energy cutoff for filtering interaction energies '
                            '(kcal/mol) prior to correlation calculation. If an interaction energy time series '
                            'absolute average value is below this cutoff, that interaction energy will not be '
                            'taken into account in correlation calculations. Defaults to 1 kcal/mol.'
                            'Applied to grinn -calc <arguments> and grinn -corr <arguments> calls.')

    parser.add_argument('--corrprefix', type=str, nargs=1, default=[''],
                        help='Prefix to the file names for storing calculation results.')

    parser.add_argument('--outfolder', default=[os.path.join(os.getcwd(),
                                                             'grinn_output')], type=str, nargs=1,
                        help='Folder path for storing calculation results. If not specified, a folder named '
                             'grinn_output will be created in the current working folder. Applies only to grinn '
                             '-calc <arguments> calls.')

    parser.add_argument('--resume', action='store_true', default=False, 
                        help='When this flag is given, gRINN will look for params.pkl in the specified OUTFOLDER. '
                            'If params.pkl is present, calculation will resume from either after initial filtering or the filtering step. '
                            'If params.pkl is not present in output folder, this flag is ignored.')

    parser.add_argument('--version', action='store_true', default=False,
                        help='Prints the version number.')

    # Parse arguments.
    args = parser.parse_args()

    calcMode = args.calc
    corrMode = args.corr
    resultsMode = args.results

    # Check which mode is requested. Either one is selected or none is selected to
    # enter GUI mode.
    print('calcMode,corrMode,resultsMode',calcMode,corrMode,resultsMode)
    if [calcMode, corrMode, resultsMode].count(True) > 1:
        print('You should specify either -calc or -corr or specify none of them '
              'to enter the GUI mode.')
        #sys.exit(0)
    elif [calcMode, corrMode].count(True) == 1:
        if calcMode:
            # User requested command-line calculation of interaction energies.
            getResIntEn(args)
        elif corrMode:
            # User requested command-line calculation of interaction energy correlations.
            import corr
            corr.getResIntCorr(args, logFile=None)
    else:
        if args.version:
            # User requested printing of version.
            versionfile = open(common.resource_path('VERSION'), 'r')
            versionline = versionfile.readlines()
            version = versionline[0].rstrip('\n')
            print(version)

        else:
            # Start the GUI.
            def startGUI(mode):
                
                from PyQt5 import QtGui
                from PyQt5 import QtCore
                from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox
                import resultsGUI, calcGUI, grinnGUI_design, calc, corr, common

                # The following two lines is required to stop multiprocessing pool childs
                # from freezing/stalling (this happens sometimes when they try to use
                # logging library. more info: https://pythonspeed.com/articles/python-multiprocessing/)
                #from multiprocessing import set_start_method
                #set_start_method("spawn")

                import sys, time, os, argparse, multiprocessing, subprocess

                multiprocessing.freeze_support()  # Required for Windows compatibility, harmless for Unix.

                class DesignInteract(QMainWindow, grinnGUI_design.Ui_gRINN):

                    def __init__(self, parent=None):
                        super(DesignInteract, self).__init__(parent)
                        self.setupUi(self)

                        _translate = QtCore.QCoreApplication.translate
                        self.label_3.setText(_translate("gRINN",
                                                        "<html><head/><body><p><a href=\"http://grinn.readthedocs.io/en/latest/tutorial.html\"><span style=\" font-size:20pt; text-decoration: underline; color:#0000ff;\">Tutorial</span></a></p></body></html>"))
                        self.label_4.setText(_translate("gRINN",
                                                        "<html><head/><body><p><a href=\"http://grinn.readthedocs.io/en/latest/credits.html\"><span style=\" font-size:20pt; text-decoration: underline; color:#0000ff;\">Credits</span></a></p></body></html>"))
                        self.label_5.setText(_translate("gRINN",
                                                        "<html><head/><body><p><a href=\"http://grinn.readthedocs.io/en/latest/contact.html\"><span style=\" font-size:20pt; text-decoration: underline; color:#0000ff;\">Contact</span></a></p></body></html>"))
                        self.label_6.setText(_translate("gRINN",
                                                        "<html><head/><body><p><a href=\"http://grinn.readthedocs.io/en/latest/faq.html\"><span style=\" font-size:20pt; text-decoration: underline; color:#0000ff;\">FAQ</span></a></p></body></html>"))

                        self.label_3.setOpenExternalLinks(True)
                        self.label_4.setOpenExternalLinks(True)
                        self.label_5.setOpenExternalLinks(True)
                        self.label_6.setOpenExternalLinks(True)
                        self.pushButton.clicked.connect(self.calculateGUI)
                        self.pushButton_2.clicked.connect(self.resultsGUI)

                    def calculateGUI(self):
                        self.formGetResIntEnGUI = calcGUI.DesignInteractCalculate(self)
                        self.formGetResIntEnGUI.show()
                        icon = QtGui.QIcon()
                        pixmap = QtGui.QPixmap(common.resource_path(
                            os.path.join('resources', 'clover.ico')))
                        icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                        self.formGetResIntEnGUI.setWindowIcon(icon)
                        self.formGetResIntEnGUI.label_3.setPixmap(pixmap)

                    def resultsGUI(self):
                        self.formResults = resultsGUI.DesignInteractResults(self)
                        self.formResults.show()
                        icon = QtGui.QIcon()
                        pixmap = QtGui.QPixmap(common.resource_path(
                            os.path.join('resources', 'clover.ico')))
                        icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                        self.formResults.setWindowIcon(icon)
                        # Skip through tab widgets to show each GUI component
                        # (apparently necessary for plots to draw correctly...
                        self.formResults.tabWidget.setCurrentIndex(0)
                        self.formResults.tabWidget.setCurrentIndex(2)
                        self.formResults.tabWidget.setCurrentIndex(3)
                        self.formResults.tabWidget.setCurrentIndex(4)
                        self.formResults.tabWidget.setCurrentIndex(5)
                        self.formResults.tabWidget_2.setCurrentIndex(0)
                        self.formResults.tabWidget_2.setCurrentIndex(1)
                        self.formResults.tabWidget_2.setCurrentIndex(0)
                        self.formResults.tabWidget.setCurrentIndex(0)
                        time.sleep(1)
                        folderLoaded = self.formResults.updateOutputFolder()

                    # if not folderLoaded:
                    #   self.formResults.close()

                    def closeEvent(self, event):
                        message = False
                        closeCalc = False
                        # Check whether any calcGUI or resultsGUI views have been created.
                        if hasattr(self, "formGetResIntEnGUI"):
                            if self.formGetResIntEnGUI.isVisible():
                                message = 'At least one "New Calculation" interface is active. Are you sure ?'
                                closeCalc = True
                        if hasattr(self, "formResults"):
                            if self.formResults.isVisible():
                                message = 'At least one "View Results" interface is active. Are you sure ?'

                        if message:
                            # Is the user sure about this?
                            buttonReply = QMessageBox.question(
                                self, 'Are you sure?', message,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                            if buttonReply == QMessageBox.No:
                                event.ignore()
                            elif buttonReply == QMessageBox.Yes:
                                if closeCalc:
                                    self.formGetResIntEnGUI.close()
                                    event.accept()
                                else:
                                    event.accept()
                        elif not message:
                            event.accept()

                def prepareEnvironment():
                    # Set some environment variable for pyinstaller executable function.
                    filePath = os.path.abspath(__file__)
                    os.environ['FONTCONFIG_FILE'] = common.resource_path(
                        os.path.join('data', 'etc', 'fonts', 'fonts.conf'))
                    os.environ['FONTCONFIG_PATH'] = common.resource_path(
                        os.path.join('data', 'etc', 'fonts'))
                    os.environ['QT_XKB_CONFIG_ROOT'] = common.resource_path(
                        os.path.join('data', 'xkb'))
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = common.resource_path(
                        os.path.join('PyQt5', 'Qt', 'plugins', 'platforms'))
                    os.environ['QT_PLUGIN_PATH'] = common.resource_path(
                        os.path.join('PyQt5', 'Qt', 'plugins', 'platforms'))

                prepareEnvironment()
                # os.environ['LD_LIBRARY_PATH'] = common.resource_path(
                #   os.path.join('data','xkb'))
                # print(os.environ['LD_LIBRARY_PATH'])
                # print(os.environ['QT_PLUGIN_PATH'])

                def main():
                    sys_argv = sys.argv
                    sys_argv += ['--style', 'Fusion']
                    app = QApplication(sys_argv)
                    form = DesignInteract()
                    icon = QtGui.QIcon()
                    pixmap = QtGui.QPixmap(common.resource_path(
                        os.path.join('resources', 'clover.ico')))
                    icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                    form.setWindowIcon(icon)
                    form.label.setGeometry(QtCore.QRect(50, 10, 161, 151))
                    form.label.setText("")
                    form.label.setScaledContents(True)
                    form.label.setPixmap(pixmap)
                    form.show()
                    app.exec_()

                if mode=='results':
                    resultsGUI.main()
                elif mode=='main':
                    main()

            if resultsMode:
                startGUI(mode='results')
            else:
                startGUI(mode='main')