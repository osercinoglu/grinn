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
	
class DesignInteract(QtWidgets.QMainWindow,design.Ui_MainWindow):
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
		self.pushButton_Stop.clicked.connect(self.stopCalculation)

		# for getResIntEn process
		self.pid = None
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
		name,__ = QtWidgets.QFileDialog.getExistingDirectory(self,'Select',os.getcwd())
		self.lineEdit_outputFolder.setText(name)

	def updateNAMDPath(self):
		name,__ = QtWidgets.QFileDialog.getOpenFileName(self,'Select',os.getcwd())
		self.lineEdit_namd2.setText(name)

	def incrementFilteringProgressBar(self,percent):
		if percent > self.progressBar_filtering.value():
			self.progressBar_filtering.setValue(percent)

	def incrementCalculationProgressBar(self,percent):
			self.progressBar_calculation.setValue(percent)

	def done(self):
			self.resetProgressElements()
			QtWidgets.QMessageBox.information(self,"Done!","Done with computation!")

	def error(self,message):
			self.monitorProgressThread.exit()
			self.resetProgressElements()
			QtWidgets.QMessageBox.information(self,"Error!",message)

	def stopCalculation(self):
		f = open('getResIntEn.log','w')
		f.write('ABORT')
		f.close()
		# Parse the log file for any child PID spawned by getResIntEn.py
		if self.pid:

			# Reset all progress elements.
			self.resetProgressElements()

			f = open(self.params.logFile,'r')
			lines = f.readlines()
			f.close()

			# Find out all the processes started by getResIntEn.py
			start_pids = list()
			complete_pids = list()
			for line in lines:
				startpid_match = re.match('.*Started a pairwise energy calculation chunk with PID: (\d+),(\d+)',line)
				completepid_match = re.match('.*Completed a pairwise energy calculation chunk with PID: (\d+),(\d+)',line)
				if startpid_match:
					start_pid = int(startpid_match.groups()[0])
					start_pid_vmd = int(startpid_match.groups()[1])
					start_pids.append(start_pid)
					start_pids.append(start_pid_vmd)

				if completepid_match:
					complete_pid = int(completepid_match.groups()[0])
					complete_pid_vmd = int(completepid_match.groups()[1])
					complete_pids.append(complete_pid)
					complete_pids.append(complete_pid_vmd)

			active_pids = [start_pid for start_pid in start_pids if start_pid not in complete_pids]

			#mainProcess = psutil.Process(self.pid.pid)
			#mainProcess.kill()
			#self.pid = None

			if active_pids:
				# Find the process ids of all child processes of all child processes of getResIntEn.py
				# and, of course, kill them.
				for pid in active_pids:
					try:
						parent = psutil.Process(pid)
						for child in parent.children(recursive=True):  # or parent.children() for recursive=False
							child.kill()
						parent.kill()
					except:
						continue



	def resetProgressElements(self):

		self.monitorProgressThread.stop()
		self.monitorLogThread.stop()
		self.progressBar_filtering.setValue(0)
		self.progressBar_calculation.setValue(0)
		self.pushButton_Stop.setEnabled(False)
		self.pushButton_Calculate.setEnabled(True)

		#self.progressBar_calculation.setValue(0)

		if self.pid:
			mainProcess = psutil.Process(self.pid.pid)
			mainProcess.kill()
			self.pid = None

	def startCalculation(self):

		# Get necessary input arguments.
		self.params.psfFile = self.lineEdit_psf.text()
		self.params.pdbFile = self.lineEdit_pdb.text()
		self.params.dcdFile = self.lineEdit_dcd.text()
		self.params.sourceSel = self.lineEdit_residueGroup1.text()
		self.params.targetSel = self.lineEdit_residueGroup2.text()
		self.params.pairFilterPercentage = self.doubleSpinBox_filteringPercent.value()
		self.params.pairFilterCutoff = self.doubleSpinBox_filteringCutoff.value()
		self.params.numCores = self.spinBox_numProcessors.value()
		self.params.skip = self.plainTextEdit_dcdStride.toPlainText()
		self.params.outputFolder = self.lineEdit_outputFolder.text()
		self.params.namd2exe = self.lineEdit_namd2.text()
		# Date: %d.%d.%d %d:%d \n' % (now.year,now.month,now.day,now.hour,
		now = datetime.datetime.now()
		self.params.logFile = 'getResIntEnLog_%d%d%d_%d%d%d.log' % (now.year,now.month,now.day,
			now.hour,now.minute,now.second)

		# Make the log file now.
		subprocess.call('touch %s' % self.params.logFile,shell=True)
		
		# Start calculation in the background
		getResIntEnArgs = ['--pdb',self.params.pdbFile,'--psf',self.params.psfFile,
		'--dcd',self.params.dcdFile,'--numcores',
		str(self.params.numCores),'--sourcesel',self.params.sourceSel,'--targetsel',self.params.targetSel,
		'--paircalc','--pairfiltercutoff',str(self.params.pairFilterCutoff),
		'--pairfilterpercentage',str(float(self.params.pairFilterPercentage)*0.1),
		'--skip',str(self.params.skip),'--namd2exe',self.params.namd2exe,
		'--outfolder',self.params.outputFolder,'--logfile',self.params.logFile]
		
		self.pid = subprocess.Popen(['./getResIntEn.py']+getResIntEnArgs)

		self.pushButton_Stop.setEnabled(True)
		self.pushButton_Calculate.setEnabled(False)

		# Start progress monitoring thread
		self.monitorProgressThread = monitorProgress(self,self.params)
		self.monitorProgressThread.incrementFilteringProgressBar.connect(self.incrementFilteringProgressBar)
		self.monitorProgressThread.incrementCalculationProgressBar.connect(self.incrementCalculationProgressBar)
		self.monitorProgressThread.success.connect(self.done)
		self.monitorProgressThread.start()

		# Start error monitoring thread
		self.monitorLogThread = monitorLog(self,self.params)
		self.monitorLogThread.error.connect(self.error)
		self.monitorLogThread.start()

class monitorProgress(QtCore.QThread):
	incrementFilteringProgressBar = pyqtSignal(int)
	incrementCalculationProgressBar = pyqtSignal(int)
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
		# Monitor filtering steps
		percent = 0
		while percent != 100:
			if os.path.isfile('getResIntEn.log'):
				logFile = open('getResIntEn.log','r')
				logLines = logFile.readlines()
				logFile.close()
				if logLines and logLines[0].startswith('ABORT'):
					print('Here')
					self.incrementFilteringProgressBar.emit(0)
					self.exit()
					return
				elif logLines:
					percent = int(float(logLines[0]))
					self.incrementFilteringProgressBar.emit(percent)

		# Monitor calculation
		continueFlag = False
		while not continueFlag:
			logFile = open('getResIntEn.log','r')
			logLines = logFile.readlines()
			if logLines and logLines[0].startswith('ABORT'):
				self.incrementCalculationProgressBar.emit(0)
				self.exit()
				return
			elif len(logLines) < 2:
				logFile.close()
				time.sleep(1)
			else:
				continueFlag = True

		if logLines and logLines[0].startswith('ABORT'):
			self.incrementCalculationProgressBar.emit(0)
			self.exit()
			return

		percent = 0
		while percent != 100:
			logFile = open('getResIntEn.log','r')
			logLines = logFile.readlines()
			logFile.close()
			if logLines and logLines[0].startswith('ABORT'):
				print('here again')
				self.incrementCalculationProgressBar.emit(0)
				self.exit()
				return
			elif logLines:
				numFilteredPairs = int(logLines[1])
				numCalculatedPairs = len(glob.glob(self.params.outputFolder+'/*_energies.dat'))
				percent = int(float(numCalculatedPairs)/float(numFilteredPairs)*100)
				self.incrementCalculationProgressBar.emit(percent)

		if percent == 100:
			self.success.emit()

		self.exit()

	def stop(self):
		self._isRunning = False

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
		while not errorLines:
			errorLines = self.parseLog(self.params.logFile)
			#time.sleep()

		self.error.emit(errorLines[0])
		self.exit()

	def stop(self):
		self._isRunning = False

def main():
	sys_argv = sys.argv
	sys_argv += ['--style', 'Fusion']
	app = QtWidgets.QApplication(sys.argv)
	form = DesignInteract()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()
