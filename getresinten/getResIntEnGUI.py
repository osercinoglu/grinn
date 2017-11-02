from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import sys
import design
import subprocess
import os
import glob
import datetime
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
	
class DesignInteract(QtWidgets.QMainWindow,design.Ui_MainWindow):
	stopMonitorProgressThread = pyqtSignal()
	stopMonitorLogThread = pyqtSignal()
	
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
		self.pushButton_BrowseParameterFile.clicked.connect(self.updateParameterFilePath)
		self.pushButton_Stop.clicked.connect(self.stopCalculation)
		self.checkBox_interactionCorrelation.clicked.connect(self.updateInteractionCorrelation)

		# for getResIntEn process
		self.thread = None
		self.params = getResIntEnParams()

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
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_parameterFile.setText(name)

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
		self.label_etaFiltering.setText(etaString)

	def updateETAcalculationLabel(self,etaString):
		self.label_etaCalculation.setText(etaString)

	def updateETAcorrelationLabel(self,etaString):
		self.label_etaCorrelation.setText(etaString)

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
		if self.thread:

			self.thread.send_signal(signal.SIGINT)

			# # Kill VMD PID as well via finding its pids in the log file
			# logFile = open(self.params.logFile,'r')
			# lines = logFile.readlines()
			# logFile.close()
			# vmd_active_pids = list()
			# vmd_completed_pids = list()
			# for line in lines:
			# 	active_matches = re.search(
			# 		'Started a pairwise energy calculation chunk with VMD PID: (\d+)',line)
			# 	if active_matches:
			# 		vmd_active_pids.append(active_matches.groups()[0])
			# 	completed_matches = re.search(
			# 		'Completed a pairwise energy calculation chunk with VMD PID: (\d+)',line)
			# 	if completed_matches:
			# 		vmd_completed_pids.append(completed_matches.groups()[0])

			# vmd_running_pids = [pid for pid in vmd_active_pids if pid not in vmd_completed_pids]
			# for pid in vmd_running_pids:
			# 	process = psutil.Process(int(pid))
			# 	process.kill()

			self.stopMonitorProgressThread.emit()
			self.stopMonitorLogThread.emit()

			#self.monitorProgressThread = None
			#self.monitorLogThread = None
			# Reset all progress elements.
			
			self.resetProgressElements()

			# Hard kill

			self.thread = None

	def resetProgressElements(self):

		self.progressBar_filtering.setValue(0)
		self.progressBar_calculation.setValue(0)
		self.progressBar_correlation.setValue(0)
		self.pushButton_Stop.setEnabled(False)
		self.pushButton_Calculate.setEnabled(True)
		self.label_etaFiltering.setText('Ready.')
		self.label_etaCalculation.setText('Ready.')
		self.label_etaCorrelation.setText('Ready.')

		#self.progressBar_calculation.setValue(0)

		#if self.pid:
		#	mainProcess = psutil.Process(self.pid.pid)
		#	mainProcess.kill()
		#	self.pid = None

	def startCalculation(self):

		#### TEMPORARY!!! ####
		subprocess.call('rm getResIntEn_output -R',shell=True)
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

		getResIntEnThread = threading.Thread(target=getResIntEn.getResIntEn,
			kwargs={'psf': self.params.psfFile,'pdb': self.params.pdbFile,
			'dcd': self.params.dcdFile,'numCores':self.params.numCores,
			'sourceSel': self.params.sourceSel, 'targetSel': self.params.targetSel,
			'pairCalc': True, 'pairFilterCutoff': self.params.pairFilterCutoff,
			'pairFilterPercentage': self.params.pairFilterPercentage*0.1,
			'skip': self.params.skip, 'namd2exe': self.params.namd2exe,
			'paramFile': self.params.paramFile, 'outputFolder': self.params.outputFolder,
			'logFile': self.params.logFile, 'toPickle': True,
			'resIntCorr': self.params.interactCorr,'pairFilterBasis': 'com',
			'pairFilterSkip': 1, 'frameRange': [False], 
			'resIntCorrAverageIntEnCutoff': 1})

		self.thread = getResIntEnThread
		self.thread.start()

		self.pushButton_Stop.setEnabled(True)
		self.pushButton_Calculate.setEnabled(False)

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

		print('Running now!')

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
		while percent != 100 and self._isRunning:
			if os.path.isfile('getResIntEn.log'):
				logFile = open('getResIntEn.log','r')
				logLines = logFile.readlines()
				logFile.close()
				if logLines:
					try:
						percent = int(float(logLines[0]))
						current_time = time.time()
						etaString = getETAstring(start_time,current_time,percent)
						self.updateETAfilteringLabel.emit(etaString)
						self.incrementFilteringProgressBar.emit(percent)
					except:
						pass

		# Monitor calculation
		continueFlag = False
		while not continueFlag and self._isRunning:
			logFile = open('getResIntEn.log','r')
			logLines = logFile.readlines()
			if len(logLines) < 2:
				logFile.close()
				time.sleep(1)
			else:
				continueFlag = True

		continueFlag = True

		percent = 0
		start_time = time.time()
		while percent != 100 and self._isRunning:
			logFile = open('getResIntEn.log','r')
			logLines = logFile.readlines()
			logFile.close()
			if logLines:
				numFilteredPairs = int(logLines[1])
				numCalculatedPairs = len(glob.glob(self.params.outputFolder+'/*_energies.log'))
				percent = int(float(numCalculatedPairs)/float(numFilteredPairs)*100)
				if percent > 0:
					current_time = time.time()
					etaString = getETAstring(start_time,current_time,percent)
					self.updateETAcalculationLabel.emit(etaString)
				self.incrementCalculationProgressBar.emit(percent)

		# Monitor interaction correlation calculation.
		if self.params.interactCorr:
			percent = 0
			start_time = time.time()
			while percent != 100 and self._isRunning:
				oldpercent = percent
				logFile = open(self.params.logFile)
				lines = logFile.readlines()
				logFile.close()
				for line in lines:
					matches = re.search('.*Interaction energy correlation thread calculated percentage:\s(\d+)',
						line)
					if matches:
						newpercent = float(matches.groups()[0])
						if newpercent > oldpercent:
							percent = newpercent
							current_time = time.time()
							etaString = getETAstring(start_time,current_time,percent)
							self.updateETAcorrelationLabel.emit(etaString)
							self.incrementCorrelationCalculationProgressBar.emit(percent)
							break

		if percent == 100:
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
	form = DesignInteract()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
