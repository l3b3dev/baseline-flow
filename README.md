# baseline-flow
baseline-flow
-- run with command line parameters:
--data flow_cylinder --sensor leverage_score --n_sensors 100 --epochs 140 --plotting True --name shallow_decoder_drop

## In each dataset:
* There is a text file "parm.txt" which provides the simulation parameters used to obtain the data.
* The input and output data for the network are inside of a folder named "marker_data".
* In every "marker_data" sub-folder:
   * **particle_inputs** contains the inputs of the dataset. 
     * Each column corresponds to an individual particle. 
     * The rows of a particular column represent {phi (particle volume fraction), Re (Non-dimensional velocity), x1,......,x15 (relative location of 15 nearest neighbors)}. 
     * Each column is essentially s (equation 1) in the notes. Hence, the number of rows will be 49.
   * **particle_outputs** contains the outputs of the dataset. 
     * Each column in the file corresponds to an individual particle
     * The elements of that column represent the non-dimensional force for all the markers on the surface of the particle. 
     * Each column corresponds to f (equation 2) in the notes.
   * **particle_markers** file contains the locations of the markers on the surface of a particle.
     * Not used currently
