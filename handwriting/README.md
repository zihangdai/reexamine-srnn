This directory contains the source code needed to reproduce all experiments on the IAM-OnDB dataset:



## Prerequisite

- Python 3.6 (default)
- Python 2.7
- PyTorch 1.0
- apex (https://github.com/NVIDIA/apex/)


## Data Preprocessing


- Download the "IAM-OnDB" dataset from [http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database) to `${DOWNLOAD_DIR}`

- Copy the `iamondb/train.txt` and `iamondb/valid.txt` into the decompressed file. 

- Preprocess:

  ```bash
  python2 iamondb_data.py \
  	--data_dir ${DOWNLOAD_DIR} \
  	--save_dir ${DATA_ROOT}
  ```


## Training with Different Models

Fill the '--data\_dir' argument with ${DATA_ROOT} variable in the following scripts.



#### [1] Factorized RNN

```bash
script/rnn.sh
```



#### [2] Factorized SRNN

```bash
script/srnn.sh
```



#### [3] Hierarchical RNN

```bash
script/rnn_hier.sh
```

#### [4] Hierarchical SRNN

```bash
script/srnn_hier.sh
```
