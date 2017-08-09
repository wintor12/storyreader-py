Code for paper "predicting the quality of short narratives from social media" https://www.ijcai.org/proceedings/2017/0539.pdf

## Requirements:
install pytorch http://pytorch.org/

pip install --upgrade dill six tqdm

pip install --upgrade nltk

git clone https://github.com/pytorch/text.git

cd text

python setup.py install

(optional) pip install --upgrade pycrayon



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
if preprocessed pre_word_vec:
```bash
python train.py --pre_word_vec --gpus 0
```

## Step 4: Test
```bash
python test.py --model MODEL_PATH
```
