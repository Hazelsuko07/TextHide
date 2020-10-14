# TextHide training on GLUE with PyTorch

TextHide[1] is a practical approach for privacy-preserving natural language understanding (NLU) tasks. It requires all participants in a distributed or federated learning setting to add a simple encryption step to prevent an eavesdropping attacker from recovering private text data. 

TextHide is inspired by InstaHide[2], which has achieved good performance in computer vision for privacy-preserving distributed learning, by providing a cryptographic security while incurring small utility loss and computation overhead.

This repository provides PyTorch implementation for fine-tuning BERT[3] models with TextHide on the GLUE benchmark[4].


## Citation
If you use TextHide or this code in your research, please cite our paper:
```
@inproceedings{hscla20,
 title = {TextHide: Tackling Data Privacy in Language Understanding Tasks},
 author ={Yangsibo Huang and Zhao Song and Danqi Chen and Kai Li and Sanjeev Arora},
 booktitle={The Conference on Empirical Methods in Natural Language Processing (Findings of EMNLP)},
 year={2020}
}
```

## How to run
### Install dependencies
- Create an Anaconda environment with Python3.6
```
conda create -n texthide python=3.6
```
- Run the following command to install dependencies
```
conda activate texthide
pip install -r requirements.txt
```

### Data preparation
Before training, you need to download the [GLUE data](https://gluebenchmark.com/tasks). By running the following script, the GLUE dataset will be saved under `/path/to/glue`.
```
python download_glue_data.py --data_dir /path/to/glue --tasks all
```

### Run TextHide Training
We proposed two TextHide schemes: TextHide-intra, which encrypts an input using other examples from the same dataset, and TextHide-inter, which utilizes a large public dataset to perform encryption. 

Due to a large public dataset, TextHide-inter is arguably more secure than TextHide-intra (but the latter is quite secure in practice when the training set is large).


#### Run TextHide-intra
Here is an example for running TextHide-intra with SST-2 with **m=256, k=4**, where m (`num_sigma`) is the number of masks, and k (`num_k`) is the number of representations got mixed:
```
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --dropout 0.4 \
  --num_train_epochs 20.0 \
  --num_k 4 \
  --num_sigma 256 \
  --output_dir ./results/$TASK_NAME/BERT_256_4_intra/ \
  --overwrite_output_dir
```

where the task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

#### Run TextHide-inter
To run TextHide-inter with SST-2, you can simply append `--inter` to the command, and use `--pub_set` to assign the public dataset, e.g. 

```
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --dropout 0.4 \
  --num_train_epochs 20.0 \
  --num_k 4 \
  --num_sigma 256 \
  --output_dir ./results/$TASK_NAME/BERT_256_4_inter/ \
  --inter \
  --pub_set MNLI \
  --overwrite_output_dir
```

#### Compatibility with SOTA models
This repository also provides support for RoBERTa models[5]. You you may run RoBERTa finetuning by assigning `--model_name_or_path` 'roberta-base'.

## Questions
If you have any questions, please open an issue or contact yangsibo@princeton.edu.

## Acknowledgements
This implementation is mainly based on [Transformers](https://github.com/huggingface/transformers/tree/master/examples/text-classification), a library for Natural Language Understanding (NLU) and Natural Language Generation (NLG).


## References:
[1] [**TextHide: Tackling Data Privacy in Language Understanding Tasks**](http://arxiv.org/abs/2010.06053), *Yangsibo Huang, Zhao Song, Danqi Chen, Kai Li, Sanjeev Arora*, Findings of EMNLP 2020

[2] [**InstaHide: Instance-hiding Schemes for Private Distributed Learning**](http://arxiv.org/abs/2010.02772), *Yangsibo Huang, Zhao Song, Kai Li, Sanjeev Arora*, ICML 2020

[3] [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805), *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*, NAACL-HLT 2019


[4] [**GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding**](https://arxiv.org/abs/1804.07461), *Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman*, ICLR 2019


[5] [**RoBERTa: A Robustly Optimized BERT Pretraining Approach**](https://arxiv.org/abs/1907.11692), *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov*, arXiv preprint

## Related Repositories
- [InstaHide](https://github.com/Hazelsuko07/InstaHide)