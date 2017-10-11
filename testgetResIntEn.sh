rm log.log;
rm outFolder -R;
./getResIntEn.py --pdb test.pdb --psf test.psf --dcd test.dcd --numcores 8 --targetsel all --sourcesel all --paircalc --pairfiltercutoff 12 --pairfilterpercentage 10 --skip 10 --namd2exe NAMD_2.12b1/namd2 --outfolder outFolder --logfile log.log