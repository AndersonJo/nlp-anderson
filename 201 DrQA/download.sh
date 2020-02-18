#!/bin/bash
# 페이스북 DrQA의 download.sh 파일을 참고 하였습니다.
# 때문에 다운로드하는 구조가 매우 유사합니다.
# 하지만 사용하는 라이브러리, 데이터파일은 다릅니다.
# https://github.com/facebookresearch/DrQA

DOWNLOAD_PATH="./data"
DATASET_PATH="$DOWNLOAD_PATH/datasets"
EMBEDDING_PATH="$DOWNLOAD_PATH/embeddings"
PIP="pip3"
PYTHON="python3.6"

# Install Spacy
sudo ${PIP} install spacy==2.2.3
sudo ${PIP} install spacy[cuda100]
sudo ${PYTHON} -m spacy download en
sudo ${PYTHON} -m spacy download en_core_web_sm


# Download SQuAD Dataset
wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

# Download official eval for SQuAD
curl "https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/" >  "./scripts/squad_eval.py"

# Get Embeddings
mkdir -p data/embeddings
wget  http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip -O $EMBEDDING_PATH/glove.840B.300d.zip
unzip $EMBEDDING_PATH/glove.840B.300d.zip -d $EMBEDDING_PATH/

echo "DrQA download done!"

