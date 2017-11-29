#!/usr/bin/env python3

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget
import viewResultsGUI_design
import sys
import pandas
import numpy as np
import seaborn
import matplotlib
import os
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from common import getResindex
from common import getChainResnameResnum
import getProEnNet
import networkx as nx
from prody import *
from pymolwidget import PyMolWidget
import math

class viewResultsParams(object):
	def __init__(self):
		self.system = None
		self.traj = None
		self.numFrames = None
		self.currentFrame = None
		self.outputFolder = None
		self.intEnMeanTotal = None
		self.intEnTotal = None
		self.resCorrTotal = None
		self.intEnCorrTotal = None
		self.networkRO = None
		self.selectedSourceRes = 0
		self.selectedTargetRes = 0
		self.selectedShortestPath = None

class MyMplCanvas(FigureCanvas):
	"""Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
	def __init__(self, parent=None, width=6, height=4, dpi=100,toolbar=False):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.fig = fig
		self.axes = fig.add_subplot(111)
		self.axes.clear()
		# We want the axes cleared every time plot() is called

		self.compute_initial_figure()

		FigureCanvas.__init__(self, fig)
		self.toolbar = NavigationToolbar(fig.canvas, self)
		if not toolbar:
			self.toolbar.hide()
		
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self,
			QSizePolicy.Expanding,
			QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.draw()

		def compute_initial_figure(self):
			t = np.arange(0.0, 3.0, 0.01)
			s = np.sin(2*np.pi*t)
			self.axes.plot(t, s)
			self.axes.clear()

class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)

    def update_figure(self,mainWindow,type='time-series'):

    	# Update the figure with new parameters
    	viewResultsParams = mainWindow.viewResultsParams
    	intEnTotal = viewResultsParams.intEnTotal
    	intEnMeanTotal = viewResultsParams.intEnMeanTotal
    	selectedSourceRes = viewResultsParams.selectedSourceRes
    	selectedTargetRes = viewResultsParams.selectedTargetRes
    	selSource_string = getChainResnameResnum(viewResultsParams.system,selectedSourceRes)
    	selTarget_string = getChainResnameResnum(viewResultsParams.system,selectedTargetRes)

    	# Rather complicated to select the correct dict key, but it should work.
    	key1 = selSource_string+'-'+selTarget_string
    	key2 = selTarget_string+'-'+selSource_string
    	if key1 not in intEnTotal.columns:
    		if key2 not in intEnTotal.columns:
    			s = np.zeros((len(intEnTotal)))
    			key = key2
    		else:
    			s = intEnTotal[key2]
    			key = key2
    	else:
    		s = intEnTotal[key1]
    		key = key1
    	
    	t = np.arange(0,len(intEnTotal),1)

    	self.axes.clear()

    	currentFrame = mainWindow.viewResultsParams.currentFrame

    	if type=='time-series':
    		self.axes.plot(t,s,'b',label=key if key else '')
    		self.axes.set_xlabel('Frame')
    		self.axes.set_ylabel('Total Non-bonded IE [kcal/mol]')
    		self.fig.subplots_adjust(left=0.2,right=0.95,bottom=0.1,top=0.99)

    		if currentFrame:
    			# Plot a tracer dot. Remember that in the following data currentFrame is two minus.
    			# This is because first frame is from the PDB loaded.
    			# Also python is zero-indexed.
    			self.axes.plot(currentFrame-2,s[currentFrame-2],linestyle=None,marker='o',color='red')

    	elif type=='distribution':
    		seaborn.kdeplot(s,ax=self.axes)
    		self.axes.set_xlabel('Total Non-bonded IE [kcal/mol]')
    		self.axes.set_ylabel('Kernel Density')
    		self.fig.subplots_adjust(left=0.2,right=0.95,bottom=0.1,top=0.99)

    	elif type=='bar-plot':
    		data = pandas.DataFrame(columns=['res','en'])
    		data['res'] = np.arange(0,len(intEnMeanTotal),1)
    		data['en'] = intEnMeanTotal[selectedSourceRes,:]
    		data_nonzero = data[data['en'] != np.float64(0)]
    		res = data_nonzero['res'].values
    		en = data_nonzero['en'].values
    		# Mind that the following does not work in mpl2.1.x
    		self.axes.barh(bottom=np.arange(0,len(res),1),width=en,color="b")
    		self.axes.set_yticks(np.arange(0,len(res),1))
    		self.axes.set_yticklabels([getChainResnameResnum(viewResultsParams.system,res) for res in res])
    		self.axes.set_xlabel('Mean IE [kcal/mol]')
    		self.fig.subplots_adjust(left=0.45,right=0.95,bottom=0.1,top=0.99)

    	elif type in ['iem','rc']:
    		if type=='iem':
    			matrix = intEnMeanTotal
    			vmax = 10
    			vmin = -10
    			cmap = seaborn.color_palette("BrBG", 10)

    		elif type=='rc':
    			matrix = viewResultsParams.resCorrTotal
    			vmax = np.max(matrix)
    			vmin = 0
    			cmap = 'Blues'

    		if len(matrix) > 100:
    			annot = False
    		else:
    			annot = True

    		hm = seaborn.heatmap(matrix,vmax=vmin,vmin=vmax,square=True,cmap=cmap,annot=annot,ax=self.axes)

    		hm.set_xticklabels(hm.get_xticklabels(), rotation = 60)
    		hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)

    		xticks = np.arange(0,viewResultsParams.system.numResidues(),10)
    		xticklabels = [getChainResnameResnum(viewResultsParams.system,xtick) for xtick in xticks]
    		self.axes.set_xticks(xticks)
    		self.axes.set_xticklabels(xticklabels,fontsize=7)
    		self.axes.set_yticks(xticks)
    		self.axes.set_yticklabels(xticklabels,fontsize=7)
    		self.axes.set_xlabel('Residue')
    		self.axes.set_ylabel('Residue')

    	# elif type == 'network':
    	# 	nx.draw_circular(viewResultsParams.networkRO)

    	elif type.startswith('network'):
    		if type == 'network-bc':
    			metric = nx.betweenness_centrality(viewResultsParams.networkRO)
    			title = 'Degree \n'
    		elif type == 'network-cc':
    			metric = nx.closeness_centrality(viewResultsParams.networkRO)
    			title = 'Closeness \n centrality'
    		elif type == 'network-degree':
    			metric = dict(nx.degree(viewResultsParams.networkRO))
    			title = 'Betweenness \n centrality'

    		chainResnameResnums = [getChainResnameResnum(viewResultsParams.system,key) for key in [key-1 for key in list(metric.keys())]]
    		# Note that the following does not work in mpl=2.1.0
    		self.axes.barh(bottom=np.arange(0,len(metric.keys()),1),width=list(metric.values()))
    		self.axes.set_yticks(np.arange(0,len(metric.keys()),1))
    		self.axes.set_yticklabels(chainResnameResnums,rotation=0,fontsize=7)
    		self.axes.set_ylim([0,len(metric.keys())])
    		self.axes.set_title(title)
    		self.fig.subplots_adjust(left=0.2,right=0.95,bottom=0.01,top=0.99)

    	# Note that the following does not work in mpl=2.0.2
    	# Works fine in mpl=2.1.0 but leaving the newest stable version out until a toolbar issue is fixed in mpl=2.2.0.
    	#self.fig.set_tight_layout({'pad':0.1})

    	self.draw()

class DesignInteractResults(QtWidgets.QMainWindow,viewResultsGUI_design.Ui_MainWindow):
	
	def __init__(self,parent=None):
		
		matplotlib.use('Qt5Agg')
		super(DesignInteractResults,self).__init__(parent)
		
		self.setupUi(self)

		#Ppopulate viewResultsParams object
		self.viewResultsParams = viewResultsParams()

		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderLabels(
			["Residue","Residue","IE [kcal/mol]"])
		
		#self.networkPlot = MyStaticMplCanvas(self.frame_NetworkPlot,width=4,height=2,
		#	dpi=100)
		#self.verticalLayout_10.addWidget(self.networkPlot)

		self.intEnBarPlot = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesBarPlot,width=5,height=4,
			dpi=100)
		self.verticalLayout_3.addWidget(self.intEnBarPlot)

		self.intEnTimeSeries = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
			dpi=100,toolbar=True)
		self.verticalLayout.addWidget(self.intEnTimeSeries)

		self.intEnDistributions = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
			dpi=100)
		self.verticalLayout.addWidget(self.intEnDistributions)

		self.intEnMeanMat = MyStaticMplCanvas(self.frame_tabIEM,width=5,height=4,dpi=100,toolbar=True)
		self.verticalLayout_5.addWidget(self.intEnMeanMat)

		self.resCorrTotalMat = MyStaticMplCanvas(self.frame_tabRC)
		self.verticalLayout_13.addWidget(self.resCorrTotalMat)

		self.degreePlot = MyStaticMplCanvas(self.frame_ResidueMetrics,width=4,height=2,
			dpi=100)
		self.horizontalLayout_4.addWidget(self.degreePlot)

		self.bcPlot = MyStaticMplCanvas(self.frame_ResidueMetrics,width=4,height=2,
			dpi=100)
		self.horizontalLayout_4.addWidget(self.bcPlot)

		self.ccPlot = MyStaticMplCanvas(self.frame_ResidueMetrics,width=4,height=2,
			dpi=100)
		self.horizontalLayout_4.addWidget(self.ccPlot)

		# Creating the PyMolWidget
		self.ProteinView = PyMolWidget()
		self.verticalLayoutProteinView = QtWidgets.QVBoxLayout(self.frame_ProteinView)
		self.verticalLayoutProteinView.addWidget(self.ProteinView)
		self.ProteinView.show()
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.ProteinView.sizePolicy().hasHeightForWidth())
		self.ProteinView.setSizePolicy(sizePolicy)
		self.ProteinView.setMaximumSize(QtCore.QSize(500,1000000))
		self.ProteinView.setMaximumSize(QtCore.QSize(1000,1000000))

		# Connect callbacks to UI elements
		self.pushButton_selectOutputFolder.clicked.connect(self.updateOutputFolder)
		self.tableWidget_sourceTargetResEnergies.cellClicked.connect(self.updatePairwiseEnergiesTable)
		self.pushButton_findShortestPaths.clicked.connect(self.findShortestPaths)
		self.tableWidget_ShortestPaths.cellClicked.connect(self.updateShortestPathsTable)

	def isFolderGood(self,folderPath):
		# Check sequentially if the folder contains the bare minimum necessary to load results
		# This is to prevent the user from selecting a totally random folder which does not contain
		# data generated by getResIntEn.py
		files = ['energies_intEnTotal.csv','energies_intEnElec.csv','energies_intEnVdW.csv',
		'energies_intEnMeanTotal.dat','energies_intEnMeanElec.dat','energies_intEnMeanVdW.dat',
		'system_dry.pdb']
		filePaths = [folderPath+'/'+x for x in files]
		for path in filePaths:
			if not os.path.exists(path):
				loadingMessage = QMessageBox.critical(self,"Error","I looked for file %s and could not find it.\n"
					"This is a required output file." % path)
				return False

		return True

	def onClick_intEnBarPlot(self,event):
		# Get the axis included in the intEnBatPlot MplStaticCanvas object
		axes = self.intEnBarPlot.fig.gca()

		# Get the label (residue) that is clicked on
		yticklabels = axes.yaxis.get_ticklabels()

		# Note that the event object here is different in mpl=2.1.0
		# The following only works with mpl=2.0.2
		x = event.x
		y = event.y
		xdata = event.xdata
		ydata = event.ydata
		if yticklabels and ydata:
			# Set the selectedTargetRes
			selectedTargetRes = getResindex(self.viewResultsParams.system,
				yticklabels[math.ceil(ydata)].get_text()) 

			# Don't do anything if the residue is already selected in the table
			if self.viewResultsParams.selectedTargetRes == selectedTargetRes:
				return
			
			self.viewResultsParams.selectedTargetRes = selectedTargetRes

			# Take action as if the sourceTargetRes table second column is clicked
			self.intEnTimeSeries.update_figure(self,'time-series')
			self.intEnDistributions.update_figure(self,'distribution')
			self.updateProteinResiduePairs()

	def onClick_intEnMeanMat(self,event):

		if not event.dblclick:
			return

		# Note that the event object here is different in mpl=2.1.0
		# The following only works with mpl=2.0.2
		x = event.x
		y = event.y
		xdata = event.xdata
		ydata = event.ydata

		# Get the axis included in intEnMeanMat MplStaticCanvas object
		axes = self.intEnMeanMat.fig.gca()

		# Get the indices clicked on.
		selectedSourceRes = math.ceil(xdata)
		selectedTargetRes = math.ceil(ydata)
		
		# Don't do anything is both selected source and target residues are the same with
		# previously selected ones.
		if selectedSourceRes == self.viewResultsParams.selectedSourceRes and \
			selectedTargetRes == self.viewResultsParams.selectedTargetRes:
			return

		self.viewResultsParams.selectedSourceRes = selectedSourceRes
		self.viewResultsParams.selectedTargetRes = selectedTargetRes

		self.updateProteinResiduePairs()

	def updateOutputFolder(self):
		name = str(QtWidgets.QFileDialog.getExistingDirectory(self,'Select an output folder containing energy calculation results.',os.getcwd()))
		if name:
			isFolderGood = self.isFolderGood(name)
			if not isFolderGood:
				return False
			loadingMessage = QMessageBox.about(self,"Loading...","Please click OK to start loading your data...")
			self.lineEdit_selectOutputFolder.setText(name)
			self.viewResultsParams.outputFolder = name
			self.viewResultsParams.system = parsePDB(self.viewResultsParams.outputFolder+'/system_dry.pdb')
			self.viewResultsParams.intEnMeanTotal = np.loadtxt(
				self.viewResultsParams.outputFolder+'/energies_intEnMeanTotal.dat')
			self.viewResultsParams.intEnTotal = pandas.read_csv(
				self.viewResultsParams.outputFolder+'/energies_intEnTotal.csv')
			self.viewResultsParams.networkRO,_ = getProEnNet.getProEnNet(inFolder=
				self.viewResultsParams.outputFolder)

			intEnCorrTotalPath = self.viewResultsParams.outputFolder+'/energies_resIntCorr.csv'
			if os.path.exists(intEnCorrTotalPath):
				self.viewResultsParams.intEnCorrTotal = pandas.read_csv(intEnCorrTotalPath)

			resCorrTotalPath = self.viewResultsParams.outputFolder+'/energies_resCorr.dat'
			if os.path.exists(resCorrTotalPath):
				self.viewResultsParams.resCorrTotal =np.loadtxt(resCorrTotalPath)

			# Clearing matplotlib canvases
			if hasattr(self,"intEnBarPlot"):
				self.intEnBarPlot.setParent(None)

			self.intEnBarPlot = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesBarPlot,width=5,height=4,
				dpi=100)
			self.verticalLayout_3.addWidget(self.intEnBarPlot)

			if hasattr(self,"intEnTimeSeries"):
				self.intEnTimeSeries.setParent(None)

			self.intEnTimeSeries = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
				dpi=100,toolbar=True)
			self.verticalLayout.addWidget(self.intEnTimeSeries)

			if hasattr(self,"intEnDistributions"):
				self.intEnDistributions.setParent(None)

			self.intEnDistributions = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
				dpi=100)
			self.verticalLayout.addWidget(self.intEnDistributions)

			if hasattr(self,"intEnMeanMat"):
				self.intEnMeanMat.setParent(None)

			self.intEnMeanMat = MyStaticMplCanvas(self.frame_tabIEM,width=5,height=4,dpi=100,toolbar=True)
			self.verticalLayout_5.addWidget(self.intEnMeanMat)

			if hasattr(self,"resCorrTotalMat"):
				self.resCorrTotalMat.setParent(None)

			self.resCorrTotalMat = MyStaticMplCanvas(self.frame_tabRC)
			self.verticalLayout_13.addWidget(self.resCorrTotalMat)

			self.populateGUI()

			# Connect some callbacks that need to be connected only once some data is loaded.
			self.intEnBarPlot.fig.canvas.mpl_connect('button_press_event',self.onClick_intEnBarPlot)
			self.intEnMeanMat.fig.canvas.mpl_connect('button_press_event',self.onClick_intEnMeanMat)
			return True
		else:
			return False

	def populateGUI(self):

		numResidues = len(self.viewResultsParams.intEnMeanTotal)
		self.tableWidget_sourceTargetResEnergies.setRowCount(numResidues)
		self.tableWidget_sourceTargetResEnergies.setColumnCount(3)
		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderLabels(["Residue","Residue","IE [kcal/mol]"])
		self.tableWidget_sourceTargetResEnergies.setSortingEnabled(False)
		header = self.tableWidget_sourceTargetResEnergies.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

		for i in range(0,numResidues):
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,0,QtWidgets.QTableWidgetItem(getChainResnameResnum(
					self.viewResultsParams.system,i)))
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,1,QtWidgets.QTableWidgetItem(getChainResnameResnum(
					self.viewResultsParams.system,i)))

		self.updatePairwiseEnergiesTable(0,0)

		#self.networkPlot.update_figure(self,'network')

		bc = nx.betweenness_centrality(self.viewResultsParams.networkRO)
		numBC = len(bc)
		newFrameSize = 10*numBC
		width = self.frame_ResidueMetrics.size().width()
		self.frame_ResidueMetrics.setMinimumSize(QtCore.QSize(width, newFrameSize))
		self.update()
		
		self.degreePlot.update_figure(self,'network-degree')
		self.bcPlot.update_figure(self,'network-bc')
		self.ccPlot.update_figure(self,'network-cc')

		self.comboBox_SourceResidue.clear()
		self.comboBox_TargetResidue.clear()
		chainResnameResnums = [getChainResnameResnum(self.viewResultsParams.system,i) for i in range(0,numResidues)]
		self.comboBox_SourceResidue.addItems(chainResnameResnums)
		self.comboBox_TargetResidue.addItems(chainResnameResnums)

		self.tableWidget_ShortestPaths.setColumnCount(1)
		self.tableWidget_ShortestPaths.setHorizontalHeaderLabels(["Path"])
		header = self.tableWidget_ShortestPaths.horizontalHeader()
		header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)

		self.intEnMeanMat.update_figure(self,'iem')

		self.ProteinView.loadMolFile(self.viewResultsParams.outputFolder+'/system_dry.pdb')
		trajPath = self.viewResultsParams.outputFolder+'/traj_dry.dcd'
		self.ProteinView._pymol.idle()
		self.ProteinView._pymol.draw()
		self.ProteinView._pymolProcess()
		self.ProteinView.show()

		# Load trajectory if it exists in the output folder!
		# Usually it should exist!
		if os.path.exists(trajPath):
			buttonReply = QMessageBox.question(self, 'Trajectory found', "A trajectory exists in your output folder. Would you like to load it as well?\n"
			"Warning: This might slow down the display significantly if the trajectory file size is large.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
			if buttonReply == QMessageBox.No:
				self.horizontalSlider.setEnabled(False)
				pass
			elif buttonReply == QMessageBox.Yes:
				self.horizontalSlider.setEnabled(True)
				self.viewResultsParams.traj = trajPath
				self.ProteinView._pymol.cmd.load_traj(trajPath)
				self.viewResultsParams.numFrames = self.ProteinView._pymol.cmd.count_frames()
				self.horizontalSlider.setMinimum(1)
				self.horizontalSlider.setMaximum(self.viewResultsParams.numFrames-1) # Remember first frame is already the PDB
				self.horizontalSlider.sliderMoved.connect(self.updateFrame)

		if isinstance(self.viewResultsParams.resCorrTotal,np.ndarray):
			self.resCorrTotalMat.update_figure(self,'rc')

	def updateFrame(self):
		# Set the current frame. Don't forget that first frame is the PDB file, so we assign plus one here.
		self.viewResultsParams.currentFrame = self.horizontalSlider.value()+1

		self.ProteinView._pymol.cmd.set_frame(self.viewResultsParams.currentFrame)
		self.ProteinView._pymol.cmd.refresh()
		self.ProteinView.glDraw()
		self.ProteinView._pymolProcess()
		if self.viewResultsParams.selectedTargetRes:
			self.intEnTimeSeries.update_figure(self,'time-series')

	def updateProteinResiduePairs(self):
		# Get selected two residues.
		selectedSourceRes = self.viewResultsParams.selectedSourceRes
		selectedTargetRes = self.viewResultsParams.selectedTargetRes
		source_string = getChainResnameResnum(self.viewResultsParams.system,selectedSourceRes)
		target_string = getChainResnameResnum(self.viewResultsParams.system,selectedTargetRes)

		# Select all and reset the view
		#self.ProteinView._pymol.cmd.select('all','all')
		self.ProteinView._pymol.cmd.show_as('cartoon','all')
		self.ProteinView._pymol.cmd.color('green','all')
		self.ProteinView._pymol.cmd.label('all','')
		self.ProteinView._pymol.cmd.set('cartoon_transparency','0.6')
		self.ProteinView._pymol.cmd.show_as('sticks','resi '+source_string[4:]+' and chain '+source_string[0])
		self.ProteinView._pymol.cmd.show_as('sticks','resi '+target_string[4:]+' and chain '+target_string[0])
		self.ProteinView._pymol.cmd.color('red','resi '+source_string[4:]+' and chain '+source_string[0])
		self.ProteinView._pymol.cmd.color('red','resi '+target_string[4:]+' and chain '+target_string[0])
		self.ProteinView._pymol.cmd.set('label_size','-2')
		self.ProteinView._pymol.cmd.label('resi '+source_string[4:]+' and chain '+source_string[0]+' and name ca','"%s%s%s" % (chain,resn,resi)')
		self.ProteinView._pymol.cmd.label('resi '+target_string[4:]+' and chain '+target_string[0]+' and name ca','"%s%s%s" % (chain,resn,resi)')
		self.ProteinView._pymolProcess()

	def updateProteinShortestPaths(self,path):
		self.ProteinView._pymol.cmd.show_as('cartoon','all')
		self.ProteinView._pymol.cmd.color('green','all')
		self.ProteinView._pymol.cmd.label('all','')
		self.ProteinView._pymol.cmd.set('cartoon_transparency','0.6')
		path_split = path.split('-')
		self.ProteinView._pymol.cmd.set('label_size','-2')
		for res in path_split:
			res_pymol_select = 'resi '+res[4:]+' and chain '+res[0]
			self.ProteinView._pymol.cmd.show_as('sticks',res_pymol_select)
			self.ProteinView._pymol.cmd.color('red',res_pymol_select)
			self.ProteinView._pymol.cmd.label(res_pymol_select+' and name ca','"%s%s%s" % (chain,resn,resi)')

		self.ProteinView._pymolProcess()

	def updatePairwiseEnergiesTable(self,row,column):
		numResidues = len(self.viewResultsParams.intEnMeanTotal)

		self.tableWidget_sourceTargetResEnergies.setRowCount(numResidues)
		# if the cell is a source residue
		if column == 0:
			selectedSourceRes = row

			self.viewResultsParams.selectedSourceRes = selectedSourceRes
			for i in range(0,numResidues):
				self.tableWidget_sourceTargetResEnergies.setItem(
					i,2,QtWidgets.QTableWidgetItem(
						str(self.viewResultsParams.intEnMeanTotal[
							self.viewResultsParams.selectedSourceRes,i])))
				self.tableWidget_sourceTargetResEnergies.item(i,0).setBackground(QtGui.QColor(255,255,255))

			# change background color to let user remember which sourceres was selected
			self.tableWidget_sourceTargetResEnergies.item(row,0).setBackground(QtGui.QColor(100,100,150))

			# plot bar plot of all other interactions with this residue
			self.intEnBarPlot.update_figure(self,'bar-plot')
		# if the cell is a target residue
		elif column in [1,2]:
			selectedTargetRes = row
			self.viewResultsParams.selectedTargetRes = selectedTargetRes
			self.intEnTimeSeries.update_figure(self,'time-series')
			self.intEnDistributions.update_figure(self,'distribution')
			
			# Update the protein structure
			self.updateProteinResiduePairs()

	def findShortestPaths(self):
		sourceRes_string = str(self.comboBox_SourceResidue.currentText())
		targetRes_string = str(self.comboBox_TargetResidue.currentText())

		if sourceRes_string and targetRes_string:
			sourceRes_index = getResindex(self.viewResultsParams.system,sourceRes_string)
			targetRes_index = getResindex(self.viewResultsParams.system,targetRes_string)
			try:
				allpaths = list(nx.all_shortest_paths(self.viewResultsParams.networkRO,
				sourceRes_index+1,targetRes_index+1))
			except:
				QtWidgets.QMessageBox.information(self,"No Paths!","No paths found!")
				return

			if allpaths:
				allpaths_string = [[getChainResnameResnum(self.viewResultsParams.system,
					int(index)-1) for index in path] for path in allpaths]
				self.tableWidget_ShortestPaths.setRowCount(len(allpaths_string))

				for i in range(0,len(allpaths_string)):
					path_string = allpaths_string[i]
					self.tableWidget_ShortestPaths.setItem(
						i,0,QtWidgets.QTableWidgetItem('-'.join(path_string)))

	def updateShortestPathsTable(self,row,column):
		selectedPath = str(self.tableWidget_ShortestPaths.item(row,0).text())
		self.updateProteinShortestPaths(selectedPath)

def main():
	sys_argv = sys.argv
	sys_argv += ['--style', 'Fusion']
	app = QtWidgets.QApplication(sys.argv)
	app.setWindowIcon(QtGui.QIcon(sys.path[0]+'/clover.ico'));
	#app.setStyle(QtWidgets.QStyleFactory.create('Macintosh'))
	form = DesignInteractResults()
	# Directly call output folder selection dialog.
	form.show() # First, show the form, somehow in some systems OpenGL requires to be activated via
	# showing to the user!

	#Skip through tab widgets to show each GUI component (apparently necessary for plots to draw correctly...
	form.tabWidget.setCurrentIndex(0)
	form.tabWidget.setCurrentIndex(2)
	form.tabWidget.setCurrentIndex(3)
	form.tabWidget.setCurrentIndex(4)
	form.tabWidget.setCurrentIndex(5)
	form.tabWidget_2.setCurrentIndex(0)
	form.tabWidget_2.setCurrentIndex(1)
	form.tabWidget_2.setCurrentIndex(2)
	form.tabWidget_2.setCurrentIndex(0)
	form.tabWidget.setCurrentIndex(0)

	folderLoaded = form.updateOutputFolder()
	if folderLoaded:
		app.exec_()
	else:
		app.exec_()
		#form.close()
		#app.quit()

if __name__ == '__main__':
	main()
