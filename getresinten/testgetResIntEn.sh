rm log.log;
rm outFolder -R;
./getResIntEn.py --pdb ../test/test.pdb --psf ../test/test.psf --dcd ../test/test.dcd --numcores 8 --targetsel resid 0 to 50 --sourcesel resid 0 to 50 --paircalc --pairfiltercutoff 12 --pairfilterpercentage 10 --skip 100 --namd2exe ../NAMD_2.12b1/namd2 --outfolder outFolder --logfile log.log --resintcorr --topickle
