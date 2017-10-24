# A script to evaluate residue-based energies from a trajectory sequentially.
# This script is basically a namdEnergy wrapper.

# Prepare
package require namdenergy
package require readcharmmpar

# Parse arguments
puts "Running"

# psf and dcd file names are by default the first two arguments.
set namd2exe [lindex $argv 0]
set outputFolder [lindex $argv 1]
set psfFile [lindex $argv 2]
set dcdFile [lindex $argv 3]

# if a skip is specified, this will be the fifth argument.
if {[llength $argv] > 4} { set skip [lindex $argv 4] } else { set skip 1}

#### THIS IS USELESS BUT ALSO HARMLESS NOW.... ###
# if an output folder is specified, this will be the fifth argument.
#if {[llength $argv] > 4} { set outputFolder [lindex $argv 4] } else { set outputFolder [pwd]}
#################################################

# frame range is by default the 6th and 7th argument
if {[llength $argv] > 5} { set frameRange [lrange $argv 5 6] } else { set frameRange "all"}

# parameter file argument is by default the 8th argument
set defpar [file join $env(CHARMMPARDIR) par_all27_prot_lipid_na.inp]
if {[llength $argv] > 7} { set paramFile [lindex $argv 7] } else { set paramFile $defpar}
if { $paramFile == "False"} { set paramFile $defpar }

# the rest of the arguments must be pairs for which interactions are to be calculated
if {[llength $argv] > 8} { set pairResidues [lrange $argv 8 end]}
set numPairResidues [llength $pairResidues]
#set halfNumPairResidues [expr{$numPairResidues/2}]
#puts $numPairResidues
#puts $halfNumPairResidues



# Add the DCD file and wait until all frames are loaded.
mol new $psfFile
if { [lindex $frameRange 1] == -1} {
	mol addfile $dcdFile waitfor all step $skip	
} else {
	mol addfile $dcdFile waitfor all step $skip first [lindex $frameRange 0] last [lindex $frameRange 1]
}

for {set i 0} {$i < $numPairResidues} {incr i 2} {

	set resid1 [lindex $pairResidues $i]
	set resid2 [lindex $pairResidues [expr {$i+1}]]

	set selEnergy1 [atomselect top "residue $resid1"]
	set selEnergyCA1 [atomselect top "residue $resid1 and name CA"]
	set selSegname1 [$selEnergyCA1 get segname]
	set selResid1 [$selEnergyCA1 get resid]
	set selResidue1 [$selEnergyCA1 get residue]

	set selEnergy2 [atomselect top "residue $resid2"]
	set selEnergyCA2 [atomselect top "residue $resid2 and name CA"]
	set selSegname2 [$selEnergyCA2 get segname]
	set selResid2 [$selEnergyCA2 get resid]
	set selResidue2 [$selEnergyCA2 get residue]

 	# Uncomment the following three lines if you want to have output files named according to segname and resid.
	#set oFileString $outputString\_$selSegname\_$selResid\_$sel2Segname\_$sel2Resid\_energies.dat
	#puts "Calculating the non-bonded interaction energy between $selSegname $selResid and $sel2Segname $sel2Resid ..."
	#namdenergy -nonb -sel $selEnergy $selEnergy2 -ofile $oFileString -tempname $rd\_$trd -skip $skip

	# Uncomment the following three lines if you want to have output files named according to residue.
	set oFileString $outputFolder\/$selResidue1\_$selResidue2\_energies.dat
	#puts "Calculating the non-bonded interaction energy between $selResidue1 and $selResidue2 ..."
	
	#namdenergy -nonb -sel $selEnergy1 $selEnergy2 -ofile $oFileString -tempname $selResidue1\_$selResidue2 -exe $namd2exe
	namdenergy -nonb -sel $selEnergy1 $selEnergy2 -ofile $oFileString -tempname $selResidue1\_$selResidue2 -exe $namd2exe -par $paramFile
	
}
	
#eval runNamdEnergyPerResidue $argv
exit
