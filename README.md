# Music-KB-AI

The main code to run our system is in test.py.  Simply run python3 test.py at the shell and follow the prompts given.

classify_kb.py contains code for all of the machine learning experiments that we did as well as the functions that predict the key of the chords the user inputs.

music_kb.py contains functions to read in and compare our two datasets.

NOTE: all of our files are included in the git repository except for msd_keys.txt which is too large (this contains song/key data for the entire MSD).

Three of the .pkl files are required to run our system, namely, svm_major.pkl, svm_minor.pkl, and svm_absolute.pkl.  These are our classifier models.