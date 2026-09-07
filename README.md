# ECHI-SpeechEnhancement

This documentation is currently very brief and will be updated in the near future.

## Setting Up

This repo relies on astral `uv` for environment management. The environments are split between training and evaluation due to conflicting dependencies. To set up and activate the environments, run

```[bash]
source setup
source start train # for the training environment
source start eval # for the eval environment
```

Also ensure that you have the [CHiME-9 ECHI data](https://huggingface.co/datasets/CHiME9-ECHI/CHiME9-ECHI) downloaded and unpacked into `data/chime9_echi` (the `chime9_echi` folder will be created during untarring). Also ensure you have a folder `data/scratch` for the data unpacking and experiment directories.

## Running

To train a model, do

```[bash]
python3 run_train.py shared.exp_name=<EXPERIMENT_NAME>
```

On the first run, this will run an unpacking script which prepares the data into `data/scratch`, so training will not start immediately. 

To enhance and evaluate a model, run
```[bash]
python3 run_enhancement.py shared.exp_name=<EXPERIMENT_NAME>
python3 run_analysis.py shared.exp_name=<EXPERIMENT_NAME>
```

## Citation

If you use this work, please cite as 
```[bib]
@inproceedings{sutherland2026challenges,
author = {Sutherland, Robert and Goetze, Stefan and Barker, Jon},
title = {Challenges of Multi-Speaker Extraction for Real Conversational Speech},
booktitle = {Proceedings of the 19th International Workshop on Acoustic Signal Enhancement (IWAENC 2026)},
year = {2026},
address = {Cremona, Italy},
url = {https://iwaenc2026.org/program/poster_97.html}
}
```
