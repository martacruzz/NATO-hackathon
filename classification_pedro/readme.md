# Project organization

## Python files:

1. **rar_to_spec.py**: this files turns .rar files into spectrograms (.png and .npy), however, there seems to be a mismatch between the output of this file with the data we were given.

2. **detection**: simple detection program, rule based and cannot tell apart drones. Seems to work best with our own processed data (from the raw files) instead of using the provided.

3. **train.py**: trains a model very slowly for the data of the *DroneRFb-Spectra*, it may be outdated.

4. **train_speed.py**: trains a model for the data of the *DroneRFb-Spectra*, the difference is that it tries to use all possible computing resources.

5. **test.py**: give it a .npy file and it tries to see which drone it correlates to. Can also be given a dataset.


