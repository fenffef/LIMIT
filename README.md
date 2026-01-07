# LIMIT
Learning from Mistakes: Self-correct Adversarial Training for Chinese Unnatural
Text Correction

## Dependencies

Main libraries

- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [transformers](https://github.com/huggingface/transformers) 4.21.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 1.0.0beta

```
pip install transformers==4.21.0
pip install fastNLP==1.0.0beta
pip install sortedcontainers
pip install nltk
pip install pyrouge
pip install rouge
pip install sentencepiece
pip install protobuf==3.20.1
```


All code only supports running on Linux.


### Datasets

we have prepared the raw dataset to help you reproduce the results in our paper.  Datasets provided by this repo can  **only**  be used for *Reproduction* and *Research Purposes*.
All files are in `jsonl` format where each line is a `json` sample:

```
{"source": "ben次比赛甴浙江省口老年体协主办，温州市老年ti协承办", "target": "本次比赛由浙江省老年体协主办，温州市老年体协承办"}
```

You can Download the jsonl files through this [link](https://drive.google.com/file/d/1H3lStuaCZnsaXW5aLOaD2xSsL1abL5EB/view?usp=drive_link).

Before loading the training set, please pre-tokenize these files  with the following command:

```
mkdir jsonl_files
mkdir tokenized_files
mv /download/path/limit_data.zip  ./jsonl_files
cd jsonl_files
unzip limit_data.zip && cd ..
python preprocess/preprocess.py --model_name  t5-limit-base --dataset hybrid
```

This command will produce the tokenized files of Hybrid `tokenized_files/train.t5.jsonl, tokenized_files/val.t5.jsonl` with the tokenizer of t5-limit-base

### Training

We have provided the training scripts for each dataset we used in this paper, and you can easily start the training process with them:

```
conda activate cont
#If there is no warmed-up checkpoint, you should use `--warmup True` to train the generation model with NLLLoss 
CUDA_VISIBLE_DEVICES=0,1 python run_hybrid.py --mode train --gpus 0,1 --warmup False --model_name t5-limit-base
```

the warmed-up checkpoint will be saved to `./pretrained_weigths/hybrid/t5` by default.  
We also provide finetuned checkpoints.  You can Download the checkpoint files through this [link](https://drive.google.com/file/d/1XFVRZU6dF7NpIX7LQ-Kp0wKa75PTU9c7/view?usp=sharing).

```
#If you already have a warmed-up checkpoint, you can skip the warm-up stage
python run_hybrid.py --mode train --gpus 0,1 --warmup False
```

After completing the training process,  several best checkpoints will be stored in a folder named after the training start time by default, for example, `checkpoints/hybrid/t5/2024-08-05-10-37-24-196200`

### Generation

We suggest first selecting the best checkpoint based on the dev set with `--mode val` and then generating the results on the test set with the best checkpoint. 

You can run the following command to generate the results on test/dev set with all checkpoints in a given folder, e.g., `checkpoints/hybrid/t5/2024-08-05-10-37-24-196200/`:

```
python run_hybrid.py --mode val --model_name bart-base-chinese --save_path checkpoints/hybrid/t5/2024-08-05-10-37-24-196200/ --gpus 0,1
```

### Evaluation

We have proveded the evaluation scripts for each dataset: `evaluation/$dataset/eval.py` with which you can easily get the evaluation results.

This is an example to evaluate all the generated results for `hybrid` in the folder `results/hybrid/t5/2024-08-05-10-37-24-196200`:

```
python evaluation/correct/eval.py --sys_path results/hybrid/t5/2024-08-05-10-37-24-196200
```
