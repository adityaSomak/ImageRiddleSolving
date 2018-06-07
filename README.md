# Solving Image Riddles

   This repository contains the code with respect to the results presented in [Answering Image Riddles using Vision and Reasoning through Probabilistic Soft Logic](https://arxiv.org/abs/1611.05896). The updated
   version of this work is accepted in UAI 2018 (Title: Combining Knowledge and Reasoning through Probabilistic Soft Logic for Image Puzzle Solving).

## Pre-requisities

  - Create a python virtual-environment, activate it and install packages using the provided `requirements.txt`.
  - In the same virtual environment, follow the instructions in `conceptnet_setup_script.sh` to set up ConceptNet5 and its association-space matrix version.

## Riddles Dataset: Pre-Processing

### All Seed Collection from Clarifai Files

 - Follow the following steps to store the ground-truth names of the riddles in a file.
    - `python findFiles.py $ARG0 > $OP0`
    - copy the file `$OP0` to `$ARG0`
    - copy `seeds.sh` to `$ARG0`
    - execute `./seeds.sh $OP0`

### Directories SETUP:

1. `conceptnet_util.py` requires:
   - The association-space representation of ConceptNet 5: `../conceptnet5/data/assoc/assoc-space-5.4`
   - Word-Vectors trained on Google News corpus: `../../../DATASETS/GoogleNews-vectors-negative300.bin`
2. `mergeTargets.py` Requires the following directory setup.
```
intermediateFiles/allTargets/test1/;
|
|_aardvark__targets__sorted.txt
|_aardvark__targets_.txt
|_...
```

### Targets Pre-processing for Fast ConceptNet access:

Firing ConceptNet queries for each word for each image in a riddle is quiet expensive in terms of running time. Hence, we pre-process
and store the most similar targets for each ground-truth answer in the riddles dataset.

1. Base Form Creation: To create the base forms, run:
```
python preprocess/targetsFileBaseWords.py ../First250riddles/riddlesDetectionAll/seeds3333_for_conceptnet.txt > intermediateFiles/lemmatizedSeeds_all3k.txt
python preprocess/EigenvectorCentrality.py intermediateFiles/lemmatizedSeeds_all3k.txt > intermediateFiles/lemmatizedSeedsCentralized_all3k.txt
```

2. Then Sort Suggested Targets:
```
 .scripts/sortSuggestedTargets.sh intermediateFiles/allTargets/test1
```

### Run Inference

1. For ResidualNetwork detections, run:
 ```
 python testAllStagesGUR.py intermediateFiles/lemmatizedSeedsCentralizedResNet_3k.txt ../First250riddles/riddlesDetectionAll/ 3333
 <outputFolder> resnet -stage all
 ```
2. For Clarifai detections, run:
```
python testAllStagesGUR.py intermediateFiles/lemmatizedSeedsCentralizedResNet_3k.txt ../First250riddles/riddlesDetectionAll/ 3333  <outputFolder> clarifai -stage all
usage: testAllStagesGUR.py [-h] [-stage STAGE] [-from FROM] [-to TO]
                           [-par PAR]
                           seedsCentralityfile detectionsFolder numPuzzles
                           inferenceFolder {clarifai,resnet}
STAGE: all/merge/clarifai
```
Choose `stage` as `all` for running the entire pipeline, `clarifai` for running the greedy solution upto visual detections and `merge` for running upto rank and retrieve stage.

3. For running UR and IUR (termed as BUR in paper) variants, just use `testAllStagesUR.py` and `testAllStagesIUR.py` respectively.


## Calculating Post-Run Accuracy:

```
usage: postrun/calculatePostRunAccuracy.py [-h] [-cleanup CLEANUP]
                                   [-summaryFile SUMMARYFILE]
                                   [-ignoreDevDataFile IGNOREDEVDATAFILE]
                                   inferenceFolder maxOrAvg
python postrun/calculatePostRunAccuracy.py intermediateFiles/resnet/output_iur_merge_pc1_r/ max -ignoreDevDataFile
```


### Gitlab Commit:
```
git status
git add *.py
git commit -m "pushing riddle code" *.py
git push -u origin master
```

## Disclaimer

   This is one of the first python-based software I wrote. So, a complete overhaul is required to increase the readability of the code.
    However, given the pre-processing steps are done correctly, this code should be used to re-produce the results presented in the Image Riddle paper.