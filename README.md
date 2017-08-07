Code for paper "predicting the quality of short narratives from social media" https://www.ijcai.org/proceedings/2017/0539.pdf

## Quickstart

## Step 1: Prepare the data
```bash
python prepare.py
```
All data are in the data folder. 

## Step 2: Preprocess
```bash
python preprocess.py --pre_word_vec WORD_VEC_PATH
```

## Step 3: Train
```bash
python train.py --pre_word_vec (if you preprocessed wordvec) --gpus 0
```

## Step 4: Test
```bash
python test.py --model MODEL PATH
```
