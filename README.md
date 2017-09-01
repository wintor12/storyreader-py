Code for paper "predicting the quality of short narratives from social media" https://www.ijcai.org/proceedings/2017/0539.pdf

## Requirements:
install pytorch http://pytorch.org/

pip install --upgrade dill six tqdm

pip install --upgrade nltk

git clone https://github.com/pytorch/text.git

cd text

python setup.py install

pip install --upgrade pycrayon



## Quickstart

## Step 1: Prepare the data
All data must be in the data folder. 
```bash
python prepare.py
```

## Step 2: Preprocess
Fix story length to 360
```bash
python preprocess.py
```
Flexible length:
```bash
python preprocess.py --fix_length 0
```
Use pretrained word vectors
```bash
python preprocess.py --pre_word_vec WORD_VEC_PATH
```

## Step 3: Train
Flexible length:
```bash
python train.py --region_nums 0 --gpus 0
```

If preprocessed pre_word_vec:
```bash
python train.py --pre_word_vec --gpus 0
```

## Step 4: Test
```bash
python test.py --model MODEL_PATH
```

===============================================

# Separate training
## Train view and log view feature:
```bash
python train_feature.py --gpus 0
python train_feature.py --gpus 0 --mode pred --trained_model feature_model/MODEL
```
if want to use random forest, use --model RF
## Train text:
```bash
python preprocess.py --fix_length 0 --data ./story_model/ --text
python train.py --region_nums 0 --gpus 0 --r_emb 50 --text 
python test.py --model story_model/MODEL --gpu 1 --data story_model/
```
## Train together:
```bash
python preprocess.py --fix_length 0 --data ./residual_model/
python residual_train.py --f_model feature_model --t_model text_model --gpus 0 --data residual_model/
```
## Train with fixed word embedding
```bash
python preprocess.py --pre_word_vec WORDVEC_PATH --fix_length 0 --text --data story_model/
python train.py --reader h --text --pre_word_vec --fix_word_vec --region_nums 0 --r_emb 50 --data story_model/ --save story_model/FOLDER/ --gpus 0
python residual_train.py --f_model feature_model/MODEL --t_model story_model/MODEL --gpus 2 --data residual_model/ --save residual_model/FOLDER/
```
## Train with combined word embedding
```bash
python preprocess.py --text --pre_word_vec WORDVEC1 --pre_word_vec2 WORDVEC2 --fix_length 0 --data story_model/ --save_word_vec_name wv_combined.pt
python train.py --reader h --text --pre_word_vec wv_combined.pt --fix_word_vec --region_nums 0 --r_emb 50 --data story_model/ --save story_model/FOLDER/ --gpus 2
python residual_train.py --f_model feature_model/MODEL --t_model story_model/MODEL --gpus 2 --data residual_model/ --save residual_model/FOLDER/
