This directory contains the source code needed to reproduce all experiments on the MIDI datasets:

- MUSE
- Nottingham



## Prerequisite

- Python 3.6
- PyTorch 1.0



## Data Preprocessing

```bash
# Firstly, set the "DATA_ROOT"
DATA_ROOT=./

python midi_data.py --save_dir ${DATA_ROOT}
```



## Main Comparison with Different Models

#### [1] Factorized RNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 571736 (0.57M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 100 \
	--d_data 88 \
	--model_name rnn
```



#### [2] Factorized SRNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 2280792 (2.28M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 100 \
	--d_data 88 \
	--model_name srnn
```



#### [3] Hierarchical RNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 1865217 (1.87M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 100 \
	--d_data 88 \
	--n_layer 2 \
	--model_name rnn_hier_inp
```



#### [4] Hierarchical SRNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 3047937 (3.05M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 100 \
	--d_data 88 \
	--model_name srnn_hier_inp
```



#### [5] Flat RNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 1579777 (1.58M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 8800 \
	--d_data 1 \
	--n_layer 3 \
	--model_name rnn
```



#### [6] Flat SRNN

```bash
DATASET="muse" or "nottingham"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 2236161 (2.24M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 8800 \
	--d_data 1 \
	--model_name srnn
```

