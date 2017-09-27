from PyQt4 import QtGui
from PyQt4 import QtCore
import sys
import design
import subprocess
import os
import glob
import datetime
import time

class getResIntEnParams(object):
	def __init__(self):
		self.psf = None
		self.pdb = None
		self.dcd = None
		self.sourceSel = None
		self.targetSel = None
		self.pairFilterPercentage = None
		self.prePairFilterCutoff = None
		self.pairFilterCutoff = None
		self.numCores = None
		self.skip = None
		self.outputFolder = None
		self.namd2exe = None
		self.logFile = None

class DesignInteract(QtGui.QMainWindow,design.Ui_MainWindow):
	def __init__(self,parent=None):
		super(DesignInteract,self).__init__(parent)
		self.setupUi(self)
		
		# Connect callbacks to UI elements
		self.pushButton_Calculate.clicked.connect(self.startCalculation)
		self.pushButton_browsePDB.clicked.connect(self.updatePDBPath)
		self.pushButton_browsePSF.clicked.connect(self.updatePSFPath)
		self.pushButton_browseDCD.clicked.connect(self.updateDCDPath)
		self.pushButton_browseOutputFolder.clicked.connect(self.updateOutputFolder)
		self.pushButton_BrowseNAMD.clicked.connect(self.updateNAMDPath)

	def updatePDBPath(self):
		name = QtGui.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_pdb.setText(name)

	def updatePSFPath(self):
		name = QtGui.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_psf.setText(name)

	def updateDCDPath(self):
		name = QtGui.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_dcd.setText(name)

	def updateOutputFolder(self):
		name = QtGui.QFileDialog.getExistingDirectory(self,'Select',os.getcwd())
		self.lineEdit_outputFolder.setText(name)

	def updateNAMDPath(self):
		name = QtGui.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_namd2.setText(name)

	def incrementFilteringProgressBar(self,percent):
		if percent > self.progressBar_filtering.value():
			self.progressBar_filtering.setValue(percent)

	def incrementCalculationProgressBar(self,percent):
			self.progressBar_calculation.setValue(percent)

	def done(self):
			self.resetProgressElements()
			QtGui.QMessageBox.information(self,"Done!","Done with computation!")

	def error(self,message):
			self.monitorProgressThread.exit()
			self.resetProgressElements()
			QtGui.QMessageBox.information(self,"Error!",message)

	def resetProgressElements(self):
		self.progressBar_filtering.setValue(0)
		self.progressBar_calculation.setValue(0)
		self.pushButton_Stop.setEnabled(False)
		self.pushButton_Calculate.setEnabled(True)

	def startCalculation(self):

		# Start a parameters object
		params = getResIntEnParams()

		# Get necessary input arguments.
		params.psfFile = self.lineEdit_psf.text()
		params.pdbFile = self.lineEdit_pdb.text()
		params.dcdFile = self.lineEdit_dcd.text()
		params.sourceSel = self.lineEdit_residueGroup1.text()
		params.targetSel = self.lineEdit_residueGroup2.text()
		params.pairFilterPercentage = self.doubleSpinBox_filteringPercent.value()
		params.prePairFilterCutoff = self.doubleSpinBox_preFilteringCutoff.value()
		params.pairFilterCutoff = self.doubleSpinBox_filteringCutoff.value()
		params.numCores = self.spinBox_numProcessors.value()
		params.skip = self.plainTextEdit_dcdStride.toPlainText()
		params.outputFolder = self.lineEdit_outputFolder.text()
		params.namd2exe = self.lineEdit_namd2.text()
		# Date: %d.%d.%d %d:%d \n' % (now.year,now.month,now.day,now.hour,
		now = datetime.datetime.now()
		params.logFile = 'getResIntEnLog_%d%d%d_%d%d%d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)

		# Make the log file now.
		subprocess.call('touch %s' % params.logFile,shell=True)
		
		# Start calculation in the background
		subprocess.call('./getResIntEn.py --pdb %s \
			--psf %s --dcd %s --numcores %s --sourcesel %s --targetsel %s \
			--prepaircalc --prepairfiltercutoff %s --paircalc --pairfilterbasis ca \
			--pairfiltercutoff %s \
			--pairfilterskip 100 --pairfilterpercentage %s --skip %s --topickle \
			--namd2exe %s --outfolder %s --logfile %s &' % \
			(params.pdbFile,params.psfFile,params.dcdFile,params.numCores,
				params.sourceSel,params.targetSel,params.prePairFilterCutoff,
				params.pairFilterCutoff,str(float(params.pairFilterPercentage)*0.1),
				params.skip,params.namd2exe,params.outputFolder,params.logFile),shell=True)

		self.pushButton_Stop.setEnabled(True)
		self.pushButton_Calculate.setEnabled(False)

		# Start progress monitoring thread
		self.monitorProgressThread = monitorProgress(self,params)
		self.connect(self.monitorProgressThread,QtCore.SIGNAL("incrementFilteringProgressBar(int)"),
			self.incrementFilteringProgressBar)
		self.connect(self.monitorProgressThread,QtCore.SIGNAL("incrementCalculationProgressBar(int)"),
			self.incrementCalculationProgressBar)
		self.connect(self.monitorProgressThread,QtCore.SIGNAL("success()"),self.done)
		self.monitorProgressThread.start()

		# Start error monitoring thread
		self.monitorLogThread = monitorLog(self,params)
		self.connect(self.monitorLogThread,QtCore.SIGNAL("error(PyQt_PyObject)"),self.error)
		self.monitorLogThread.start()

class monitorProgress(QtCore.QThread):

	def __init__(self,mainWindow,params):
		QtCore.QThread.__init__(self)
		self.mainWindow = mainWindow
		self.params = params

	def __del__(self):
		self.wait()

	def run(self):
		# Start monitoring the progress of computation
		subprocess.call('rm getResIntEn.log',shell=True)
		self.mainWindow.progressBar_filtering.setValue(0)
		self.mainWindow.progressBar_calculation.setValue(0)

		# Monitor filtering steps
		percent = 0
		while percent != 100:
			if os.path.isfile('getResIntEn.log'):
				logFile = open('getResIntEn.log','r')
				logLines = logFile.readlines()
				if logLines:
					percent = int(float(logLines[0]))
					self.emit(QtCore.SIGNAL('incrementFilteringProgressBar(int)'),percent)

		# Monitor calculation
		continueFlag = False
		while not continueFlag:
			logFile = open('getResIntEn.log','r')
			logLines = logFile.readlines()
			if len(logLines) < 2:
				logFile.close()
				time.sleep(1)
			else:
				continueFlag = True

		logFile = open('getResIntEn.log','r')
		logLines = logFile.readlines()
		numFilteredPairs = int(logLines[1])
		percent = 0
		while percent != 100:
			numCalculatedPairs = len(glob.glob(self.params.outputFolder+'/*_energies.dat'))
			percent = int(float(numCalculatedPairs)/float(numFilteredPairs)*100)
			self.emit(QtCore.SIGNAL('incrementCalculationProgressBar(int)'),percent)

		self.emit(QtCore.SIGNAL('success()'))

class monitorLog(QtCore.QThread):

	def __init__(self,mainWindow,params):
		QtCore.QThread.__init__(self)
		self.mainWindow = mainWindow
		self.params = params

	def __del__(self):
		self.wait()

	def parseLog(self,logFile):
		# Parse the log file for errors,mainly.
		errorFlag = False
		while errorFlag == False:
			f = open(logFile,'r')
			lines = f.readlines()
			f.close()
			errorLines = [line for line in lines if 'ERROR' in line]
			if errorLines:
				errorFlag = True

		return errorLines

	def run(self):
		# Start monitoring the progress log file produced by getResIntEn.py
		errorLines = list()
		while not errorLines:
			errorLines = self.parseLog(self.params.logFile)
			#time.sleep()

		self.emit(QtCore.SIGNAL('error(PyQt_PyObject)'),errorLines[0])

def main():
	app = QtGui.QApplication(sys.argv)
	form = DesignInteract()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
