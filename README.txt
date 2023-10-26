Refer to src/calc_classes.py to see how each calculation is 
intialized and what objects are needed start the run.

It is up to the user to extract from their ES code all the 
necessary parameters for the desired calculation, so this
package is kept generalzied for any ES code.

Place all your extraction logic in extract.sh, and read in 
those parameters in main.py. After the calculation is set up
in main.py, call the associated run() function. This will
return the updated nuclear coordinates for the next step.

The run.sh script is the sbatch script. The number of steps 
you want to perform is controlled by the loop in run.sh. Writing 
the updated coordinates to a new input file for your ES code can 
be done in either main.py or within run.sh. Cleanup unnecessary
files after a run using cleanup.sh.
