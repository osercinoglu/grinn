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
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class viewResultsParams(object):
	def __init__(self):
		self.outputFolder = None
		self.intEnMeanTotal = None
		self.intEnTotal = None
		self.selectedSourceRes = 0
		self.selectedTargetRes = 0

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
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
    	intEnTotal = viewResultsParams.intEnTotal
    	selectedSourceRes = viewResultsParams.selectedSourceRes
    	selectedTargetRes = viewResultsParams.selectedTargetRes
    	key = str((selectedSourceRes+1,selectedTargetRes+1))
    	if key not in intEnTotal.columns:
    		s = np.zeros((len(intEnTotal)))
    	else:
    		s = intEnTotal[key]
    	
    	t = np.arange(0,len(intEnTotal),1)

    	if type=='time-series':
    		self.axes.plot(t,s,'b',label=key)
    		self.axes.set_xlabel('Frame')
    		self.axes.set_ylabel('Total Non-bonded IE [kcal/mol]')

    	elif type=='distribution':
    		seaborn.kdeplot(s,ax=self.axes)
    		self.axes.set_xlabel('Total Non-bonded IE [kcal/mol]')
    		self.axes.set_ylabel('Kernel Density')

    	self.draw()

class DesignInteract(QtWidgets.QMainWindow,viewResultsGUI_design.Ui_MainWindow):
	
	def __init__(self,parent=None):
		super(DesignInteract,self).__init__(parent)
		self.setupUi(self)

		self.viewResultsParams = viewResultsParams()
		self.viewResultsParams.outputFolder = \
		'/home/onur/Dropbox/experiments/getResIntEnPlugin/2017_10_26_ChemViaCompSymposium/getResIntEn_1K5N_ABC_1'
		self.viewResultsParams.intEnMeanTotal = np.loadtxt(
			self.viewResultsParams.outputFolder+'/energies_intEnMeanTotal.dat')
		self.viewResultsParams.intEnTotal = pandas.read_csv(
			self.viewResultsParams.outputFolder+'/energies_intEnTotal.csv')

		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderItem(0,
			QtWidgets.QTableWidgetItem('Residue'))
		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderItem(1,
			QtWidgets.QTableWidgetItem('Residue'))
		self.tableWidget_sourceTargetResEnergies.setHorizontalHeaderItem(2,
			QtWidgets.QTableWidgetItem('IE [kcal/mol]'))
		# Creating matplotlib canvases 
		self.intEnTimeSeries = MyStaticMplCanvas(self.frame_tabPairwiseEnergiesPlots,width=5,height=4,
			dpi=100)
		self.verticalLayout_3.addWidget(self.intEnTimeSeries)
		self.intEnDistributions = MyStaticMplCanvas(self.frame_tabPairwiseEnergiesPlots,width=5,height=4,
			dpi=100)
		self.verticalLayout_3.addWidget(self.intEnDistributions)

		# Connect callbacks to UI elements

		self.populateTable()

		self.tableWidget_sourceTargetResEnergies.cellClicked.connect(self.updateTable)

	def populateTable(self):
		numResidues = len(self.viewResultsParams.intEnMeanTotal)
		self.tableWidget_sourceTargetResEnergies.setRowCount(numResidues)
		self.tableWidget_sourceTargetResEnergies.setColumnCount(3)
		for i in range(0,numResidues):
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,0,QtWidgets.QTableWidgetItem(str(i+1)))
			self.tableWidget_sourceTargetResEnergies.setItem(
				i,1,QtWidgets.QTableWidgetItem(str(i+1)))

		self.updateTable(0,0)

	def updateTable(self,row,column):
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
		# if the cell is a target residue
		elif column == 1:
			selectedTargetRes = row
			self.viewResultsParams.selectedTargetRes = selectedTargetRes
			self.intEnTimeSeries.update_figure(self.viewResultsParams,'time-series')
			self.intEnDistributions.update_figure(self.viewResultsParams,'distribution')
			pass # to do plot updating an other stuff..

def main():
	sys_argv = sys.argv
	sys_argv += ['--style', 'Fusion']
	app = QtWidgets.QApplication(sys.argv)
	form = DesignInteract()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()