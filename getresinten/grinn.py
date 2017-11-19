#!/usr/bin/env python3
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import viewResults
import getResIntEnGUI
import grinnGUI
import sys

class DesignInteract(QtWidgets.QMainWindow,grinnGUI.Ui_gRINN):

	def __init__(self,parent=None):
		super(DesignInteract,self).__init__(parent)
		self.setupUi(self)

		self.pushButton.clicked.connect(self.getResIntEnGUI)
		self.pushButton_2.clicked.connect(self.viewResults)


	def getResIntEnGUI(self):
		self.formGetResIntEnGUI = getResIntEnGUI.DesignInteractCalculate(self)
		self.formGetResIntEnGUI.show()

	def viewResults(self):
		self.formResults = viewResults.DesignInteractResults(self)
		self.formResults.show()

def main():
	sys_argv = sys.argv
	sys_argv += ['--style','Fusion']
	app = QtWidgets.QApplication(sys_argv)
	form = DesignInteract()
	form.show()
	app.exec_()

if __name__ == '__main__':
	main()