rm log.log;
rm -R outFolder ;
./getResIntEn.py --pdb ../test/test.pdb --psf ../test/test.psf --dcd ../test/test.dcd --numcores 3 --targetsel resindex 0 to 3 --sourcesel resindex 0 to 3 --paircalc --pairfiltercutoff 12 --pairfilterpercentage 10 --skip 250 --namd2exe ../NAMD_2.12_MacOSX-x86_64-multicore/namd2 --outfolder outFolder --logfile log.log --resintcorr --topickle
