id=`qsub run1.pbs`
qsub -W depend=afterok:$id run2.pbs