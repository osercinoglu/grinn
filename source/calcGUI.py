#!/usr/bin/env python
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import sys
import calcGUI_design
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
import calc
import corr
import resultsGUI
import common
import argparse
	
class DesignInteractCalculate(QtWidgets.QMainWindow,calcGUI_design.Ui_MainWindow):
	stopMonitorProgressThread = pyqtSignal()
	
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
		self.pushButton_viewResults.clicked.connect(self.viewResultsStart)
		self.checkBox_interactionCorrelation.clicked.connect(self.updateInteractionCorrelation)

		# Sample data loading (temporary for NAR review)
		self.pushButton_loadSampleNAMDdata.clicked.connect(self.loadSampleNAMDdata)
		self.pushButton_loadSampleGMXdata.clicked.connect(self.loadSampleGMXdata)

		# for getResIntEn process
		self.processGetResIntEn = None
		self.calcParams = common.parameters()

		# Set the numCores to the maximum cpu count in this system.
		numCores = multiprocessing.cpu_count()
		self.calcParams.numCores = numCores
		self.spinBox_numProcessors.setValue(int(numCores))

	def loadSampleGMXdata(self):
		#root_path = sys.path[0]
		self.lineEdit_outputFolder.setText(os.path.join(os.getcwd(),'grinn_output'))

		self.lineEdit_namd2.setText('gmx')
		self.lineEdit_pdb.setText('../samples/test.tpr')
		self.lineEdit_psf.setText('../samples/test.top')
		self.lineEdit_dcd.setText('../samples/test_stride.xtc')

	def loadSampleNAMDdata(self):
		#root_path = sys.path[0]
		self.lineEdit_outputFolder.setText(os.path.join(os.getcwd(),'grinn_output'))

		self.lineEdit_namd2.setText('/home/onur/repos/NAMD_2.12b1/namd2')
		self.lineEdit_pdb.setText('../samples/test.pdb')
		self.lineEdit_psf.setText('../samples/test.psf')
		self.lineEdit_dcd.setText('../samples/test.dcd')
		self.lineEdit_parameterFile.setText('../samples/par_all27_prot_lipid_na.inp')

	def closeEvent(self, event):
			self.stopCalculation()
			event.accept() # let the window close

	def exitHandler(self):
		self.stopCalculation()
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
		elif state == 0:
			self.doubleSpinBox_AverageIntEnCutoff.setEnabled(False)

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

	def viewResultsStart(self):
		self.formResults = resultsGUI.DesignInteractResults(self)
		self.formResults.show()

		#Skip through tab widgets to show each GUI component (apparently necessary for plots to draw correctly...
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

	def done(self,message):
		self.resetProgressElements()
		#self.monitorProgressThread.exit()
		#self.monitorLogThread.exit()
		QtWidgets.QMessageBox.information(self,"Done!",message)

	def error(self,message):
		if hasattr(self,"monitorProgressThread"):
			self.monitorProgressThread.exit()
			self.resetProgressElements()
		QtWidgets.QMessageBox.information(self,"Error!",message)

	def stopCalculation(self):
		# Parse the log file for any child PID spawned by getResIntEn.py
		if hasattr(self,"processGetResIntEn"):

			if self.processGetResIntEn:
				# Stop the calculation
				print('killing process id '+str(self.processGetResIntEn.pid))
				os.kill(self.processGetResIntEn.pid,signal.SIGINT)

			# Stop monitoring
			self.stopMonitorProgressThread.emit()

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

	def params2parser(self,calcParams):
		# Converting a parameter object to an argument parser namespace
		args = argparse.Namespace()
		args.calc = True
		args.pdb = [calcParams.pdb]
		args.tpr = [calcParams.tpr]
		args.top = [calcParams.top]
		args.traj = [calcParams.traj]
		args.numcores = [calcParams.numCores]
		args.dielectric = [calcParams.dielectric]
		args.sel1 = [calcParams.sel1]
		args.sel2 = [calcParams.sel2]
		args.pairfiltercutoff = [calcParams.pairFilterCutoff]
		args.pairfilterpercentage = [calcParams.pairFilterPercentage]
		args.stride = [calcParams.stride]
		args.framerange = [False]
		args.exe = [calcParams.exe]
		args.parameterfile = [calcParams.parameterFile]
		args.calccorr = calcParams.calcCorr
		args.corrintencutoff = [calcParams.corrIntEnCutoff]
		args.outfolder = [calcParams.outFolder]
		return args

	def startCalculation(self):
		self.calcParams = common.parameters()
		# Get necessary input arguments.
		self.calcParams.top = str(self.lineEdit_psf.text())
		if self.lineEdit_pdb.text().endswith('.pdb'):
			self.calcParams.pdb = str(self.lineEdit_pdb.text())
		elif self.lineEdit_pdb.text().endswith('.tpr'):
			self.calcParams.tpr = str(self.lineEdit_pdb.text())

		self.calcParams.traj = str(self.lineEdit_dcd.text())
		self.calcParams.sel1 = str(self.lineEdit_residueGroup1.text())
		self.calcParams.sel2 = str(self.lineEdit_residueGroup2.text())
		self.calcParams.dielectric = float(self.doubleSpinBox_soluteDielectric.value())
		self.calcParams.pairFilterPercentage = float(self.doubleSpinBox_filteringPercent.value())
		self.calcParams.pairFilterCutoff = float(self.doubleSpinBox_filteringCutoff.value())
		self.calcParams.numCores = int(self.spinBox_numProcessors.value())
		self.calcParams.stride = int(self.spinBox_dcdStride.value())
		self.calcParams.outFolder = os.path.abspath(str(self.lineEdit_outputFolder.text()))
		self.calcParams.exe = str(self.lineEdit_namd2.text())
		self.calcParams.parameterFile = str(self.lineEdit_parameterFile.text())
		state = self.checkBox_interactionCorrelation.checkState()
		if state == 2:
			self.calcParams.calcCorr = True
		elif state == 0:
			self.calcParams.calcCorr = False
		self.calcParams.corrIntEnCutoff = float(self.doubleSpinBox_AverageIntEnCutoff.value())
		
		# Date-inclusive log file name.
		# Date: %d.%d.%d %d:%d \n' % (now.year,now.month,now.day,now.hour,
		#now = datetime.datetime.now()
		#self.params.logFile = 'getResIntEnLog_%02d%02d%02d_%02d%02d%02d.log' % (now.year,now.month,now.day,
		#	now.hour,now.minute,now.second)

		self.calcParams.logFile = os.path.join(str(self.calcParams.outFolder),'grinn.log')

		if os.path.exists(os.path.abspath(str(self.calcParams.outFolder))):
			self.error("The output folder exists. Please specify a path that does not exist."
				 " Aborting now.")
			return
		elif not os.access(os.path.abspath(
			os.path.dirname(self.calcParams.outFolder)), os.W_OK):
			self.error("Can't write to the output folder path. Do you have write access?")
			return
		
		args = self.params2parser(self.calcParams)

		# Start calculation in the background
		self.processGetResIntEn = multiprocessing.Process(target=calc.getResIntEn,
			args=(args,))

		self.processGetResIntEn.start()

		#QtWidgets.QMessageBox.information(self,"Info!","PID of process is: "+str(self.processGetResIntEn.pid))

		self.pushButton_Stop.setEnabled(True)
		self.pushButton_Calculate.setEnabled(False)
		self.pushButton_viewResults.setEnabled(False)

		# Start progress monitoring thread and connect signals
		self.monitorProgressThread = monitorProgress(self,self.calcParams)
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
		self.monitorProgressThread.error.connect(self.error)
		self.stopMonitorProgressThread.connect(self.monitorProgressThread.stop)

		self.monitorProgressThread.start()

class monitorProgress(QtCore.QThread):
	incrementFilteringProgressBar = pyqtSignal(int)
	incrementCalculationProgressBar = pyqtSignal(int)
	incrementCorrelationCalculationProgressBar = pyqtSignal(int)
	updateETAfilteringLabel = pyqtSignal(str)
	updateETAcalculationLabel = pyqtSignal(str)
	updateETAcorrelationLabel = pyqtSignal(str)
	updateStatusBar = pyqtSignal(str)
	success = pyqtSignal(str)
	error = pyqtSignal(str)

	def __init__(self,mainWindow,params):
		QtCore.QThread.__init__(self)
		self.mainWindow = mainWindow
		self.params = params
		self._isRunning = True

	#def __del__(self):
	#	self.wait()

	def run(self):

		# Start monitoring the progress of computation
		self.mainWindow.progressBar_filtering.setValue(0)
		self.mainWindow.progressBar_calculation.setValue(0)
		self._isRunning = True

		# method for ETA calculation
		def getETAstring(start_time,current_time,percent):
			# Prevent division by zero below.
			if percent == float(0):
				percent += 0.000001
			elapsed_time = current_time - start_time
			remaining_time = (100-percent)*elapsed_time/float(percent)
			remaining_time_hhmm = divmod(remaining_time,60)
			etaString = 'ETA: %i min %i sec' % (remaining_time_hhmm[0],remaining_time_hhmm[1])
			return etaString

		# Wait until the logFile is created.
		while not os.path.exists(self.params.logFile):
			time.sleep(1)

		# Monitor the log file and take action depending the line read.
		start_time = time.time()
		lastLogLine = 0
		continueFlag = False
		percent = 0

		while not continueFlag and self._isRunning:
			logFile = open(self.params.logFile)
			lines = logFile.readlines()
			logFile.close()
			for i in range(lastLogLine,len(lines)):
				line = lines[i]
				if 'DEBUG' in line: continue

				matchesFiltering = re.search('.*Filtered pairs percentage:\s(\d+)',line)
				matchesCalculation = re.search('.*Completed calculation percentage: (\d+)',line)
				matchesCorrelation = re.search('.*Interaction energy correlation calculated percentage:\s(\d+)',line)
				current_time = time.time()

				if matchesFiltering:
					percent = float(matchesFiltering.groups()[0])
					etaString = 'Filtering progress: ' + getETAstring(start_time,current_time,percent)
					self.updateETAfilteringLabel.emit(etaString)
					self.incrementFilteringProgressBar.emit(percent)					

				elif matchesCalculation:
					percent = float(matchesCalculation.groups()[0])
					etaString = 'Calculation progress: ' + getETAstring(start_time,current_time,percent)
					self.updateETAcalculationLabel.emit(etaString)
					self.incrementCalculationProgressBar.emit(percent)

				elif matchesCorrelation:
					percent = float(matchesCorrelation.groups()[0])
					etaString = 'Correlation progress: ' + getETAstring(start_time,current_time,percent)
					self.updateETAcorrelationLabel.emit(etaString)
					self.incrementCorrelationCalculationProgressBar.emit(percent)

				elif 'FINAL:' in line:
					self.success.emit(line)
					continueFlag = True
					self._isRunning = False
					self.incrementCalculationProgressBar.emit(0)
					self.incrementCorrelationCalculationProgressBar.emit(0)

				elif 'ERROR' in line:
					self.error.emit(line)
					continueFlag = True
					self._isRunning = False

				if percent == float(100):
					start_time = time.time()

				self.updateStatusBar.emit(line)

			lastLogLine = i

		self.exit()

	def stop(self):
		self._isRunning = False
		#self.terminate()

def main():
	sys_argv = sys.argv
	sys_argv += ['--style', 'Fusion']
	app = QtWidgets.QApplication(sys.argv)	
	app.setWindowIcon(QtGui.QIcon(os.path.dirname(
	os.path.dirname(os.path.abspath(__file__))),'resources','clover.ico'));
	form = DesignInteractCalculate()
	icon = QtGui.QIcon()
	pixmap = QtGui.QPixmap(os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		'resources','clover.ico'))
	icon.addPixmap(pixmap,QtGui.QIcon.Normal, QtGui.QIcon.Off)
	form.setWindowIcon(icon)
	form.label_3.setPixmap(pixmap)
	#app.aboutToQuit.connect(form.exitHandler)
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
