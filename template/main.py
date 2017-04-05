__doc__="""
********************************************************************************
******                      CE807-Assignment 1 - MAIN                     ******
******           Author: Dimitrios Andreou (dandreb@essex.ac.uk)          ******
******                         Python version: 3.6                        ******
********************************************************************************

This is the main file which loads and runs the evaluation of the results. 
Running this script will allow to inspect the configuration of the pipeline 
as well as use any already trained classifier. The data are assumed to be 
under each respective task folder (except folder for task 3). Each task 
script was executed separately and their output from the terminal was
saved in separate files.

TASKS   FITTED PIPELINES                    CONFUSION MATRICES
    1   baseline_optimised_classifiers.npz  baseline_confusion.npz
    2   improved_optimised_classifiers.npz  improved_confusion.npz

TASKS   DIRECTORY   SCRIPTS                     LOGS                      
    1   Task1/      baseline.py                 baseline_log.txt
    2   Task2/      improved.py                 improved_log.txt
    3   Task3/      analysis.py
    
Custom classes used are located at the root of the assignments directory tree 
(Assignment1/utils.py):
PipelineShow    VoteClassifier  StopWordFeature   POSFeature

"""
if __name__=='__main__':
    print(__doc__)
    import os
    import nltk as nl
    import sklearn as sk
    import numpy as np
    import scipy as sc

    print(  nl.__name__,nl.__version__,
            sk.__name__,sk.__version__,
            np.__name__,np.__version__,
            sc.__name__,sc.__version__ )
    
    os.chdir('Task1')
    if not any([ 'npz' in i.split('.') for i in os.listdir()]):
        print('No baseline saved data!\n better grab a coffee...\n\n')
        os.system('python baseline.py')
    else:
        print('Baseline results found!\n')
    
    os.chdir('../Task2')
    if not any([ 'npz' in i.split('.') for i in os.listdir()]):
        print('No improved saved data!\n better grab a much bigger coffee...\n\n')
        os.system('python improved.py')
    else:
        print('Improved results found!\n')
    
    print('Running analysis...\n\n')
    os.chdir('../Task3')
    os.system('python analysis.py')