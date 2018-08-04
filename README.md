# A Constituent-Centric Neural Architecture for Reading Comprehension

Implemented in Tensorflow, python3.6


Files:
- Core Model File: ``ccrc_model.py``
- Sub-modules:
    - question encoding: ``question_encoding.py``
    - context encoding: ``context_encoding.py``
    - attention layer:``attention_layer.py``
    - data utilities and candidate answers generation:``load_data.py``
    - answer prediction and parameter learning:``ccrc_model.py``


## Prerequisites
1. python3.6
2. tensorflow 1.0.1
## Pipeline
0. git clone https://github.com/shrshore/Constituent-Centric-Neural-Architecture-for-Reading-Comprehension
1. git clone https://github.com/allenai/bi-att-flow, enter the directory and run the ``download.sh``, you will get the glove vectors and squad dataset.
2. git clone https://github.com/stanfordnlp/treelstm and enter the directory
3. run ``./fetch_and_preprocess.sh`` to download **glove** word vectors and **stanford parser**.
4. download [squad](https://rajpurkar.github.io/SQuAD-explorer/) into the same directory of glove The filepath can be modified in ``load_data.py``
5. enter the directory of *Constituent-Centric-Neural-Architecture-for-Reading-Comprehension*
6. run ``my_main.py``. 
7. You can check the hidden value representation of root node for every question in ``logger.log`` file


