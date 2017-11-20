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

class viewResultsParams(object):
	def __init__(self):
		self.system = None
		self.outputFolder = None
		self.intEnMeanTotal = None
		self.intEnTotal = None
		self.networkRO = None
		self.selectedSourceRes = 0
		self.selectedTargetRes = 0
		self.selectedShortestPath = None

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        #self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.toolbar = NavigationToolbar(fig.canvas, self)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)

    def update_figure(self,viewResultsParams,type='time-series'):
    	# Update the figure with new parameters
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

    	if type=='time-series':
    		self.axes.plot(t,s,'b',label=key if key else '')
    		self.axes.set_xlabel('Frame')
    		self.axes.set_ylabel('Total Non-bonded IE [kcal/mol]')

    	elif type=='distribution':
    		seaborn.kdeplot(s,ax=self.axes)
    		self.axes.set_xlabel('Total Non-bonded IE [kcal/mol]')
    		self.axes.set_ylabel('Kernel Density')

    	elif type=='bar-plot':
    		data = pandas.DataFrame(columns=['res','en'])
    		data['res'] = np.arange(0,len(intEnMeanTotal),1)
    		data['en'] = intEnMeanTotal[selectedSourceRes,:]
    		data_nonzero = data[data['en'] != np.float64(0)]
    		res = data_nonzero['res'].values
    		en = data_nonzero['en'].values
    		self.axes.barh(y=np.arange(0,len(res),1),width=en,color="b")
    		self.axes.set_yticks(np.arange(0,len(res),1))
    		self.axes.set_yticklabels([getChainResnameResnum(viewResultsParams.system,res) for res in res])
    		self.axes.set_xlabel('Mean IE [kcal/mol]')

    	elif type=='iem':
    		if len(intEnMeanTotal) > 100:
    			annot = False
    		else:
    			annot = True
    		seaborn.heatmap(intEnMeanTotal,vmax=10,vmin=-10,square=True,
                        cmap=seaborn.color_palette("BrBG", 10),annot=annot,ax=self.axes)

    		xticks = np.arange(0,viewResultsParams.system.numResidues(),10)
    		xticklabels = [getChainResnameResnum(viewResultsParams.system,xtick) for xtick in xticks]
    		self.axes.set_xticks(xticks)
    		self.axes.set_xticklabels(xticklabels,fontsize=7)
    		self.axes.set_yticks(xticks)
    		self.axes.set_yticklabels(xticklabels,fontsize=7)
    		self.axes.set_xlabel('Residue')
    		self.axes.set_ylabel('Residue')

    	self.fig.tight_layout()
    	self.draw()

class DesignInteractResults(QtWidgets.QMainWindow,viewResultsGUI_design.Ui_MainWindow):
	
	def __init__(self,parent=None):
		super(DesignInteractResults,self).__init__(parent)
		self.setupUi(self)

		#Ppopulate viewResultsParams object
		self.viewResultsParams = viewResultsParams()

		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderLabels(
			["Residue","Residue","IE [kcal/mol]"])
		
		# Creating matplotlib canvases
		self.intEnBarPlot = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesBarPlot,width=5,height=4,
			dpi=100)
		self.verticalLayout_3.addWidget(self.intEnBarPlot)

		self.intEnTimeSeries = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
			dpi=100)
		self.verticalLayout.addWidget(self.intEnTimeSeries)
		self.intEnDistributions = MyStaticMplCanvas(self.frame_tabPairWiseEnergiesPlots,width=5,height=4,
			dpi=100)
		self.verticalLayout.addWidget(self.intEnDistributions)
		self.intEnMeanMat = MyStaticMplCanvas(self.frame_tabIEM,width=5,height=4,dpi=100)
		self.verticalLayout_5.addWidget(self.intEnMeanMat)

		# Creating the PyMolWidget
		self.ProteinView = PyMolWidget()
		self.horizontalLayout.addWidget(self.ProteinView)
		self.ProteinView.show()
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(1)
		sizePolicy.setVerticalStretch(1)
		sizePolicy.setHeightForWidth(self.ProteinView.sizePolicy().hasHeightForWidth())
		self.ProteinView.setSizePolicy(sizePolicy)
		self.ProteinView.setMaximumSize(QtCore.QSize(600,1000000))

		# Connect callbacks to UI elements
		self.pushButton_selectOutputFolder.clicked.connect(self.updateOutputFolder)
		#self.populateGUI()
		self.tableWidget_sourceTargetResEnergies.cellClicked.connect(self.updatePairwiseEnergiesTable)

		self.pushButton_findShortestPaths.clicked.connect(self.findShortestPaths)

		self.tableWidget_ShortestPaths.cellClicked.connect(self.updateShortestPathsTable)

	def updateOutputFolder(self):
		name = str(QtWidgets.QFileDialog.getExistingDirectory(self,'Select',os.getcwd()))
		if name:
			self.lineEdit_selectOutputFolder.setText(name)
			self.viewResultsParams.outputFolder = name
			self.viewResultsParams.system = parsePDB(self.viewResultsParams.outputFolder+'/system.pdb')
			self.viewResultsParams.intEnMeanTotal = np.loadtxt(
				self.viewResultsParams.outputFolder+'/energies_intEnMeanTotal.dat')
			self.viewResultsParams.intEnTotal = pandas.read_csv(
				self.viewResultsParams.outputFolder+'/energies_intEnTotal.csv')
			self.viewResultsParams.networkRO,_ = getProEnNet.getProEnNet(inFolder=
				self.viewResultsParams.outputFolder)
			self.populateGUI()

	def populateGUI(self):

		numResidues = len(self.viewResultsParams.intEnMeanTotal)
		self.tableWidget_sourceTargetResEnergies.setRowCount(numResidues)
		self.tableWidget_sourceTargetResEnergies.setColumnCount(3)
		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderLabels(["Residue","Residue","IE [kcal/mol]"])
		self.tableWidget_sourceTargetResEnergies.setSortingEnabled(False)
		for i in range(0,numResidues):
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,0,QtWidgets.QTableWidgetItem(getChainResnameResnum(
					self.viewResultsParams.system,i)))
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,1,QtWidgets.QTableWidgetItem(getChainResnameResnum(
					self.viewResultsParams.system,i)))

		self.updatePairwiseEnergiesTable(0,0)

		self.comboBox_SourceResidue.clear()
		self.comboBox_TargetResidue.clear()
		chainResnameResnums = [getChainResnameResnum(self.viewResultsParams.system,i) for i in range(0,numResidues)]
		self.comboBox_SourceResidue.addItems(chainResnameResnums)
		self.comboBox_TargetResidue.addItems(chainResnameResnums)

		self.tableWidget_ShortestPaths.setColumnCount(1)
		self.tableWidget_ShortestPaths.setHorizontalHeaderLabels(["Path"])

		self.intEnMeanMat.update_figure(self.viewResultsParams,'iem')

		self.ProteinView.loadMolFile(self.viewResultsParams.outputFolder+'/system.pdb')
		self.ProteinView.show()

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
		#self.ProteinView._pymol.update()

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
			self.intEnBarPlot.update_figure(self.viewResultsParams,'bar-plot')
		# if the cell is a target residue
		elif column in [1,2]:
			selectedTargetRes = row
			self.viewResultsParams.selectedTargetRes = selectedTargetRes
			self.intEnTimeSeries.update_figure(self.viewResultsParams,'time-series')
			self.intEnDistributions.update_figure(self.viewResultsParams,'distribution')
			
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
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
