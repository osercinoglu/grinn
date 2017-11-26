rm log.log;
rm outFolder -R;
echo "Without resintcorr"
../getresinten/getResIntEn.py --tpr test.tpr --top test.top --traj test_stride.xtc --numcores 8 --targetsel resindex 0 to 3 --sourcesel resindex 0 to 3 --paircalc --pairfiltercutoff 12 --pairfilterpercentage 10 --skip 100 --outfolder outFolder --logfile log.log --parameterfile False --topickle