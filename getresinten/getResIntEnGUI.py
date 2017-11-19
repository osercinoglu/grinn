from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import sys
import design
import multiprocessing
import os
import glob
import datetime
import subprocess
import time
import psutil
import re
import signal
import threading
# Below only for pyinstallers compatibility
import getResIntEn
import getResIntCorr
import common

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
		self.interactCorr = False
		self.interactCorrCutoff = None
	
class DesignInteractCalculate(QtWidgets.QMainWindow,design.Ui_MainWindow):
	stopMonitorProgressThread = pyqtSignal()
	stopMonitorLogThread = pyqtSignal()
	
	def __init__(self,parent=None):
		super(DesignInteractCalculate,self).__init__(parent)
		self.setupUi(self)

		# Connect callbacks to UI elements
		self.pushButton_Calculate.clicked.connect(self.startCalculation)
		self.pushButton_browsePDB.clicked.connect(self.updatePDBPath)
		self.pushButton_browsePSF.clicked.connect(self.updatePSFPath)
		self.pushButton_browseDCD.clicked.connect(self.updateDCDPath)
		self.pushButton_browseOutputFolder.clicked.connect(self.updateOutputFolder)
		self.pushButton_BrowseNAMD.clicked.connect(self.updateNAMDPath)
		self.pushButton_BrowseParameterFile.clicked.connect(self.updateParameterFilePath)
		self.pushButton_Stop.clicked.connect(self.stopCalculation)
		self.pushButton_viewResults.clicked.connect(self.viewResults)
		self.checkBox_interactionCorrelation.clicked.connect(self.updateInteractionCorrelation)

		# for getResIntEn process
		self.processGetResIntEn = None
		self.params = getResIntEnParams()


	def closeEvent(self, event):
			self.stopCalculation()
			event.accept() # let the window close

	def exitHandler(self):
		self.stopCalculation(self)
		os._exit(0)

	def updatePDBPath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_pdb.setText(name)

	def updatePSFPath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_psf.setText(name)

	def updateDCDPath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_dcd.setText(name)

	def updateOutputFolder(self):
		name = str(QtWidgets.QFileDialog.getExistingDirectory(self,'Select',os.getcwd()))
		self.lineEdit_outputFolder.setText(name)

	def updateNAMDPath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_namd2.setText(name)

	def updateParameterFilePath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileNames(self,'Select',os.getcwd())
		paramFiles = ' '.join(name)
		self.lineEdit_parameterFile.setText(paramFiles)

	def updateInteractionCorrelation(self):
		state = self.checkBox_interactionCorrelation.checkState()
		if state == 2:
			self.doubleSpinBox_AverageIntEnCutoff.setEnabled(True)
			self.params.interactCorr = True
		elif state == 0:
			self.doubleSpinBox_AverageIntEnCutoff.setEnabled(False)
			self.params.interactCorr = False

	def incrementFilteringProgressBar(self,percent):
		if percent > self.progressBar_filtering.value():
			self.progressBar_filtering.setValue(percent)

	def incrementCalculationProgressBar(self,percent):
		self.progressBar_calculation.setValue(percent)

	def incrementCorrelationCalculationProgressBar(self,percent):
		self.progressBar_correlation.setValue(percent)

	def updateETAfilteringLabel(self,etaString):
		self.labelFiltering.setText(etaString)

	def updateETAcalculationLabel(self,etaString):
		self.labelCalculation.setText(etaString)

	def updateETAcorrelationLabel(self,etaString):
		self.labelCorrelation.setText(etaString)

	def updateStatusBar(self,string):
		self.statusbar.showMessage(string)

	def viewResults(self):
		subprocess.call(sys.path[0]+'/viewResults.py &',shell=True)

	def done(self):
		self.resetProgressElements()
		self.monitorProgressThread.exit()
		self.monitorLogThread.exit()
		QtWidgets.QMessageBox.information(self,"Done!","Done with computation!")

	def error(self,message):
		self.monitorProgressThread.exit()
		self.resetProgressElements()
		QtWidgets.QMessageBox.information(self,"Error!",message)

	def stopCalculation(self):
		# Parse the log file for any child PID spawned by getResIntEn.py
		if self.processGetResIntEn:

			# Stop the calculation
			os.kill(self.processGetResIntEn.pid,signal.SIGINT)

			# Stop monitoring
			self.stopMonitorProgressThread.emit()
			self.stopMonitorLogThread.emit()

			# Reset all progress elements.
			self.resetProgressElements()

			# Reset calculation thread
			self.processGetResIntEn = None

	def resetProgressElements(self):

		self.progressBar_filtering.setValue(0)
		self.progressBar_calculation.setValue(0)
		self.progressBar_correlation.setValue(0)
		self.pushButton_Stop.setEnabled(False)
		self.pushButton_Calculate.setEnabled(True)
		self.pushButton_viewResults.setEnabled(True)
		self.labelFiltering.setText('Filtering progress')
		self.labelCalculation.setText('Calculation progress')
		self.labelCorrelation.setText('Correlation progress')

		#self.progressBar_calculation.setValue(0)

		#if self.pid:
		#	mainProcess = psutil.Process(self.pid.pid)
		#	mainProcess.kill()
		#	self.pid = None

	def startCalculation(self):

		#### TEMPORARY!!! ####
		# subprocess.call('rm -R getResIntEn_output',shell=True)
		# self.lineEdit_namd2.setText('/Users/onur/repos/gRINN/NAMD_2.12_MacOSX-x86_64-multicore/namd2')
		# self.lineEdit_pdb.setText('/Users/onur/repos/gRINN/test/test.pdb')
		# self.lineEdit_psf.setText('/Users/onur/repos/gRINN/test/test.psf')
		# self.lineEdit_dcd.setText('/Users/onur/repos/gRINN/test/test.dcd')
		# self.lineEdit_parameterFile.setText('/Users/onur/repos/gRINN/par_all27_prot_lipid_na.inp')
		#### TEMPORARY!!! ####

		# Get necessary input arguments.
		self.params.psfFile = self.lineEdit_psf.text()
		self.params.pdbFile = self.lineEdit_pdb.text()
		self.params.dcdFile = self.lineEdit_dcd.text()
		self.params.sourceSel = self.lineEdit_residueGroup1.text()
		self.params.targetSel = self.lineEdit_residueGroup2.text()
		self.params.pairFilterPercentage = self.doubleSpinBox_filteringPercent.value()
		self.params.pairFilterCutoff = self.doubleSpinBox_filteringCutoff.value()
		self.params.numCores = self.spinBox_numProcessors.value()
		self.params.skip = int(self.doubleSpinBox_dcdStride.value())
		self.params.outputFolder = self.lineEdit_outputFolder.text()
		self.params.namd2exe = self.lineEdit_namd2.text()
		self.params.paramFile = self.lineEdit_parameterFile.text()
		self.params.interactCorrAverageIntEnCutoff = self.doubleSpinBox_AverageIntEnCutoff.value()
		# Date: %d.%d.%d %d:%d \n' % (now.year,now.month,now.day,now.hour,
		now = datetime.datetime.now()
		self.params.logFile = 'getResIntEnLog_%02d%02d%02d_%02d%02d%02d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)

		# Make the log file now.
		subprocess.call('touch %s' % self.params.logFile,shell=True)
		
		# Start calculation in the background
		self.processGetResIntEn = multiprocessing.Process(target=getResIntEn.getResIntEn,
			kwargs={'psf':self.params.psfFile,'pdb':self.params.pdbFile,
			'dcd':self.params.dcdFile,'numCores':self.params.numCores,
			'sourceSel':self.params.sourceSel,'targetSel':self.params.targetSel,
			'pairCalc':True,'pairFilterCutoff':self.params.pairFilterCutoff,
			'pairFilterPercentage':self.params.pairFilterPercentage*0.1,
			'skip':self.params.skip,'namd2exe':self.params.namd2exe,
			'paramFile':self.params.paramFile,'outputFolder':self.params.outputFolder,
			'logFile':self.params.logFile,'toPickle':True,
			'resIntCorr':self.params.interactCorr,'pairFilterBasis':'com',
			'pairFilterSkip':1,'frameRange':[False],
			'resIntCorrAverageIntEnCutoff':1})

		self.processGetResIntEn.start()

		self.pushButton_Stop.setEnabled(True)
		self.pushButton_Calculate.setEnabled(False)
		self.pushButton_viewResults.setEnabled(False)

		# Start progress monitoring thread and connect signals
		self.monitorProgressThread = monitorProgress(self,self.params)
		self.monitorProgressThread.incrementFilteringProgressBar.connect(
			self.incrementFilteringProgressBar)
		self.monitorProgressThread.incrementCalculationProgressBar.connect(
			self.incrementCalculationProgressBar)
		self.monitorProgressThread.incrementCorrelationCalculationProgressBar.connect(
			self.incrementCorrelationCalculationProgressBar)
		self.monitorProgressThread.updateETAfilteringLabel.connect(
			self.updateETAfilteringLabel)
		self.monitorProgressThread.updateETAcalculationLabel.connect(
			self.updateETAcalculationLabel)
		self.monitorProgressThread.updateETAcorrelationLabel.connect(
			self.updateETAcorrelationLabel)
		self.monitorProgressThread.updateStatusBar.connect(
			self.updateStatusBar)
		self.monitorProgressThread.success.connect(self.done)
		self.stopMonitorProgressThread.connect(self.monitorProgressThread.stop)

		self.monitorProgressThread.start()

		# Start error monitoring thread and connect signals
		self.monitorLogThread = monitorLog(self,self.params)
		self.monitorLogThread.error.connect(self.error)
		self.stopMonitorLogThread.connect(self.monitorLogThread.stop)
		self.monitorLogThread.start()

class monitorProgress(QtCore.QThread):
	incrementFilteringProgressBar = pyqtSignal(int)
	incrementCalculationProgressBar = pyqtSignal(int)
	incrementCorrelationCalculationProgressBar = pyqtSignal(int)
	updateETAfilteringLabel = pyqtSignal(str)
	updateETAcalculationLabel = pyqtSignal(str)
	updateETAcorrelationLabel = pyqtSignal(str)
	updateStatusBar = pyqtSignal(str)
	success = pyqtSignal()

	def __init__(self,mainWindow,params):
		QtCore.QThread.__init__(self)
		self.mainWindow = mainWindow
		self.params = params
		self._isRunning = True

	#def __del__(self):
	#	self.wait()

	def run(self):

		# Start monitoring the progress of computation
		subprocess.call('rm getResIntEn.log',shell=True)
		self.mainWindow.progressBar_filtering.setValue(0)
		self.mainWindow.progressBar_calculation.setValue(0)
		self._isRunning = True

		# method for ETA calculation
		def getETAstring(start_time,current_time,percent):
			elapsed_time = current_time - start_time
			remaining_time = (100-percent)*elapsed_time/float(percent)
			remaining_time_hhmm = divmod(remaining_time,60)
			etaString = 'ETA: %i min %i sec' % (remaining_time_hhmm[0],remaining_time_hhmm[1])
			return etaString

		# Monitor filtering steps
		percent = 0
		start_time = time.time()
		lastLogLine = 0
		while percent != 100 and self._isRunning:
			oldpercent = percent
			logFile = open(self.params.logFile)
			lines = logFile.readlines()
			logFile.close()
			for i in range(0,len(lines)):
				line = lines[i]
				if 'DEBUG' in line: continue
				matches = re.search('.*Filtered pairs percentage:\s(\d+)',line)
				if matches:
					newpercent = float(matches.groups()[0])
					if newpercent > oldpercent:
						percent = newpercent
						current_time = time.time()
						etaString = 'Filtering progress: ' + getETAstring(start_time,current_time,percent)
						self.updateETAfilteringLabel.emit(etaString)
						self.incrementFilteringProgressBar.emit(percent)
						break
				if i > lastLogLine:
					self.updateStatusBar.emit(line)
					lastLogLine = i

		# Monitor calculation
		continueFlag = False
		while not continueFlag and self._isRunning:
			logFile = open(self.params.logFile)
			lines = logFile.readlines()
			logFile.close()
			for i in range(0,len(lines)):
				line = lines[i]
				if 'DEBUG' in line: continue
				matches = re.search('.*Started a pairwise energy calculation thread.',line)
				if matches:
					continueFlag = True
					self.updateStatusBar.emit(line)
					break
				if i > lastLogLine:
					self.updateStatusBar.emit(line)
					lastLogLine = i
				self.updateStatusBar.emit(line)

		percent = 0
		start_time = time.time()
		while percent != 100 and self._isRunning:
			oldpercent = percent
			logFile = open(self.params.logFile)
			lines = logFile.readlines()
			logFile.close()
			for i in range(0,len(lines)):
				line = lines[i]
				if 'DEBUG' in line: continue
				matches = re.search('.*Number of interaction pairs selected after filtering step:\s(\d+)',line)
				if matches:
					numFilteredPairs = int(matches.groups()[0])
					numCalculatedPairs = len(glob.glob(self.params.outputFolder+'/*_energies.log'))
					percent = int(float(numCalculatedPairs)/float(numFilteredPairs)*100)
					if percent > oldpercent:
						current_time = time.time()
						etaString = 'Calculation progress: ' + getETAstring(start_time,current_time,percent)
						self.updateETAcalculationLabel.emit(etaString)
						self.incrementCalculationProgressBar.emit(percent)
				if i > lastLogLine:
					self.updateStatusBar.emit(line)
					lastLogLine = i

		# Monitor interaction correlation calculation.
		if self.params.interactCorr and self._isRunning:
			percent = 0
			start_time = time.time()
			while percent != 100 and self._isRunning:
				oldpercent = percent
				logFile = open(self.params.logFile)
				lines = logFile.readlines()
				logFile.close()
				for i in range(0,len(lines)):
					line = lines[i]
					if 'DEBUG' in line: continue
					matches = re.search('.*Interaction energy correlation calculated percentage:\s(\d+)',line)
					if matches:
						newpercent = float(matches.groups()[0])
						if newpercent > oldpercent:
							percent = newpercent
							current_time = time.time()
							etaString = 'Correlation progress: ' + getETAstring(start_time,current_time,percent)
							self.updateETAcorrelationLabel.emit(etaString)
							self.incrementCorrelationCalculationProgressBar.emit(percent)
					if i > lastLogLine:
						self.updateStatusBar.emit(line)
						lastLogLine = i

		continueFlag = True
		if percent == 100 and self._isRunning:
			while continueFlag is True:
				logFİle = open(self.params.logFile)
				lines = logFile.readlines()
				logFile.close()
				for i in range(lastLogLine,len(lines)):
					line = lines[i]
					self.updateStatusBar.emit(line)
					lastLogLine = i
					if 'FINAL: ' in line:
						self.success.emit()
						self.incrementCalculationProgressBar.emit(0)
						self.incrementCorrelationCalculationProgressBar.emit(0)
						self.exit()

	def stop(self):
		self._isRunning = False
		#self.terminate()

class monitorLog(QtCore.QThread):
	error = pyqtSignal(str)

	def __init__(self,mainWindow,params):
		QtCore.QThread.__init__(self)
		self.mainWindow = mainWindow
		self.params = params
		self._isRunning = True

	#def __del__(self):
	#	self.wait()

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
		while not errorLines and self._isRunning:
			errorLines = self.parseLog(self.params.logFile)
			#time.sleep()

		if self._isRunning:
			self.error.emit(errorLines[0])
		self.exit()

	def stop(self):
		self._isRunning = False
		#self.terminate()

def main():
	sys_argv = sys.argv
	sys_argv += ['--style', 'Fusion']
	app = QtWidgets.QApplication(sys.argv)
	app.setWindowIcon(QtGui.QIcon(sys.path[0]+'/clover.ico'));
	form = DesignInteractCalculate()
	#app.aboutToQuit.connect(form.exitHandler)
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
