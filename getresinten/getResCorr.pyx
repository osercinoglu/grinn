cimport numpy as np
import numpy as np
from prody import *
import itertools
import logging
import pyprind

def getResCorr(e_corrs,pdb,corrCutoff,outName,logFile):

	################################################################
	### Construct residue-residue correlation list using Cython ####
	################################################################

	logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
		datefmt='%d-%m-%Y:%H:%M:%S',level=logging.DEBUG,filename=logFile)
	logger = logging.getLogger(__name__)
	logger.info('Started residue correlation matrix construction.')

	cdef int res0, res1, pair00, pair01, pair10, pair11, lenCorrList, lenResCombins, m, n
	cdef float pearson_r_c, RC_ij

	system = parsePDB(pdb)
	numRes = system.numResidues()
	# Get all residue combinations
	resList = np.arange(0,numRes)
	#res_combins = np.asarray(list(itertools.combinations(resList,2)))
	res_combins = list(itertools.combinations(resList,2))

	lenEcorrs = len(e_corrs)
	#lenResCombins = 1000
	lenResCombins = len(res_combins)

	cdef np.ndarray RC_ij_array = np.zeros(lenEcorrs)
	#cdef np.ndarray e_corrs, res_combins
	#cdef list resCorrList
	cdef np.ndarray resCorrs = np.zeros([numRes,numRes],dtype=np.float)
	
	progbar = pyprind.ProgBar(lenResCombins,monitor=True)

	#resCorrList = list()
	logger.info('Constructing the residue correlation matrix...')
	
	for m in xrange(lenResCombins):

		res_combin = res_combins[m]
    
		res0 = res_combin[0]
		res1 = res_combin[1]
    
		#RC_ij = 0

		for n in xrange(lenEcorrs):

			corr = e_corrs[n]
		
			pair00= corr[0]
			pair01 = corr[1]
			pair10 = corr[2]
			pair11 = corr[3]
			pearson_r_c = corr[4]
		
			if ((res0 == pair00 or res0 == pair01) and (res1 == pair10 or res1 == pair11)) or ((res1 == pair00 or res1 == pair01) and (res0 == pair10 or res0 == pair11)):
			#if (res0 in [pair00,pair01] and res1 in [pair10,pair11]) or (res1 in [pair00,pair01] and res0 in [pair10,pair11]):
				RC_ij_array[n] = pearson_r_c
			else:
				RC_ij_array[n] = 0

		if resCorrMethod == 'sum':
			RC_ij = np.sum(RC_ij_array)
		elif resCorrMethod == 'mean':
			RC_ij = np.mean(RC_ij_array)
		elif resCorrMethod == 'abs-mean':
			RC_ij = np.mean(np.abs(RC_ij_array))

		#resCorrs.append([res0,res1,RC_ij])
		resCorrs[res0,res1] = RC_ij
		resCorrs[res1,res0] = RC_ij
		
		progbar.update()

	logger.info('Constructing the residue correlation matrix... completed.')

	# Get columns sums of residue-residue correlation matrix to obtain connectivity measures for each amino acid
	resCorrSums = np.sum(resCorrs,axis=0)

	# Write resCorrSums and resCorrs to txt files
	np.savetxt('%s_resCorrSums.txt' % outname,resCorrSums)
	np.savetxt('%s_resCorrsMat.txt' % outname,resCorrs)

	# Write a pdb file with beta column of total residue correlation values
	system = parsePDB(pdb)
	
	residues = system.select('name CA')
	for i in range(0,len(residues)):
		residue = system.select('resindex %i' % i)
		resCorr = resCorrSums[i]
		residue.setBetas([resCorr]*residue.numAtoms())
	
	writePDB('%s_resCorrSumBeta.pdb' % outname,system)

	logger.info('Done.')