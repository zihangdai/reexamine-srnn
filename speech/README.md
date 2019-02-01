This directory contains the source code needed to reproduce all experiments on the speech datasets:

- TIMIT
- VCTK
- Blizzard
- Permuted TIMIT (Perm-TIMIT)



## Prerequisite

- Python 3.6
- PyTorch 1.0
- torchaudio (https://github.com/pytorch/audio)
- apex (https://github.com/NVIDIA/apex/)
- ffmpeg (for vctk and blizzard datasets) `apt install ffmpeg`


## Data Preprocessing

#### TIMIT

- Download the "TIMIT" dataset from [LDC website](https://catalog.ldc.upenn.edu/LDC93S1) to `${DOWNLOAD_DIR}`

- Decompress the file `timit_LDC93S1.tgz`: `tar -zxvf timit_LDC93S1.tgz`

- Preprocess:

  ```bash
  python timit_data.py \
  	--data_dir ${DOWNLOAD_DIR}/timit/TIMIT \
  	--save_dir ${DATA_ROOT}/timit
  ```



#### Perm-TIMIT

- `python create_permuted_data.py --data_dir DATA_ROOT/timit`

- The script above will generate the `DATA_ROOT/timit-permuted-200`



#### VCTK

- Download the "VCTK" dataset from [https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) to `${DOWNLOAD_DIR}`

- Decompress the file `unzip VCTK-Corpus.zip`.

- Preprocess:

  ```bash
  python vctk_data.py \
  	--data_dir ${DOWNLOAD_DIR}/VCTK-Corpus \
		--save_dir ${DATA_ROOT}/vctk
  ```




#### Blizzard

- Download the "Blizzard" dataset from [https://www.synsig.org/index.php/Blizzard_Challenge_2013](https://www.synsig.org/index.php/Blizzard_Challenge_2013) to `${DOWNLOAD_DIR}`

- Decompress the file

- Preprocess:

  ```bash
  python blizzard_data.py \
  	--data_dir ${DOWNLOAD_DIR}/blizzard2013/Lessac_Blizzard2013_CatherineByers_train/unsegmented \
		--save_dir ${DATA_ROOT}/blizzard
  ```


## Main Comparison with Different Models



#### [1] Factorized RNN

```bash
# For Perm-TIMIT, please also use "timit"
DATASET="timit" or "vctk" or "blizzard"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 17412912 (17.41M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 40 \
	--d_data 200 \
	--model_name rnn \
	--n_layer 2 \
	--d_rnn 750
```



#### [2] Factorized SRNN

```bash
# For Perm-TIMIT, please also use "timit"
DATASET="timit" or "vctk" or "blizzard"

# DATA_ROOT is the same path used in data preprocessing
DATA_DIR=${DATA_ROOT}/${DATASET}

# Expected param size: 17525848 (17.53M)
CUDA_VISIBLE_DEVICES=0 python train.py \
	--dataset ${DATASET} \
	--data_dir ${DATA_DIR} \
	--tgt_len 40 \
	--d_data 200 \
	--model_name srnn \
	--d_rnn 825
```



#### [3] Hierarchical RNN

- <u>TIMIT</u> with **a single GPU**:

  ```bash
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 17275736 (17.28M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_hier_inp \
  	--d_rnn 700
  ```



- <u>VCTK</u> and <u>Blizzard</u> with **four GPUs**:

  ```bash
  DATASET="vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 17275736 (17.28M)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_hier_inp \
  	--d_rnn 700
  ```

  

- <u>Perm-TIMIT</u> with **a single GPU**:

  ```bash
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}-permuted-200
  
  # Expected param size: 17094012 (17.09M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_hier_nade \
  	--d_rnn 512 \
  	--d_emb 256 \
  	--n_low_layer 3 \
  	--d_nade 64
  ```

  

#### [4] Hierarchical SRNN

- <u>TIMIT</u> with **a single GPU**:

  ```bash
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 17248352 (17.25M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name srnn_hier_inp \
  	--d_rnn 660
  ```

  

- <u>VCTK</u> and <u>Blizzard</u> with **four GPUs**:

  ```bash
  DATASET="vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 17248352 (17.25M)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_hier_inp \
  	--d_rnn 660
  ```

  

- <u>Perm-TIMIT</u> with **a single GPU**:

  ```bash
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}-permuted-200
  
  # Expected param size: 17076076 (17.08M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name srnn_hier_nade \
  	--d_rnn 536 \
  	--d_emb 256 \
  	--d_mlp 256 \
  	--d_lat 256 \
  	--n_low_layer 3 \
  	--d_nade 64
  ```

  

#### [5] Flat RNN

- <u>TIMIT</u> and <u>Perm-TIMIT</u> with **a single GPU**:

  ```bash
  # For Perm-TIMIT, please also use "timit"
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16857484 (16.86M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name rnn \
  	--n_layer 2 \
  	--d_rnn 1100
  ```

  

- <u>VCTK</u> and <u>Blizzard</u> with **two GPUs**:

  ```bash
  DATASET="vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16857484 (16.86M)
  CUDA_VISIBLE_DEVICES=0,1 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name rnn \
  	--n_layer 2 \
  	--d_rnn 1100
  ```



#### [6] Flat SRNN

- <u>TIMIT</u> and <u>Perm-TIMIT</u> with **a single GPU**:

  ```bash
  # For Perm-TIMIT, please also use "timit"
  DATASET="timit"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16930838 (16.93M)
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name srnn \
  	--n_layer 1 \
  	--d_rnn 825
  ```

  

- <u>VCTK</u> and <u>Blizzard</u> with **three GPUs**:

  ```bash
  DATASET="vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16930838 (16.93M)
  CUDA_VISIBLE_DEVICES=0,1,2 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name srnn \
  	--n_layer 1 \
  	--d_rnn 825
  ```



## Verify "Advantage under High Volatility" (section 4.3)

- RNN (two GPUs)

  ```bash
  DATASET="timit" or "vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16857484 (16.86M)
  CUDA_VISIBLE_DEVICES=0,1 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name rnn \
  	--n_layer 2 \
  	--d_rnn 1100
  	--down_sample 200
  ```

  

- SRNN (three GPUs)

  ```bash
  DATASET="timit" or "vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  # Expected param size: 16930838 (16.93M)
  CUDA_VISIBLE_DEVICES=0,1,2 python multiprocessing_train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 2000 \
  	--d_data 1 \
  	--model_name srnn \
  	--n_layer 1 \
  	--d_rnn 825 \
  	--down_sample 200
  ```

  

## Verify "Intra-step Correlation" with delta-RNN  (section 4.4)

- Interleaving split:

  ```bash
  # For Perm-TIMIT, please also use "timit"
  DATASET="timit" or "vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_interleave \
  	--n_layer 2 \
  	--d_rnn 750 \
  	--chk_len 3
  ```

  

- Random split:

  ```bash
  # For Perm-TIMIT, please also use "timit"
  DATASET="timit" or "vctk" or "blizzard"
  
  # DATA_ROOT is the same path used in data preprocessing
  DATA_DIR=${DATA_ROOT}/${DATASET}
  
  CUDA_VISIBLE_DEVICES=0 python train.py \
  	--dataset ${DATASET} \
  	--data_dir ${DATA_DIR} \
  	--tgt_len 40 \
  	--d_data 200 \
  	--model_name rnn_random \
  	--n_layer 2 \
  	--d_rnn 750 \
  	--d_leak 75
  ```

  
