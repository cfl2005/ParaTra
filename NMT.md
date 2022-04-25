# ChineseNMT

This is a **Transformer** based neural machine translation(**NMT**) modelðŸ¤—.

## Data

The dataset is from [WMT 2018 Chinese-English track](http://statmt.org/wmt18/translation-task.html) (Only NEWS Area)

## Data Process

### Word Segmentation

- **Tool** sentencepiece](https://github.com/google/sentencepiece)
- **Preprocess** un `./data/get_corpus.py` , in which we will get bilingual data to build our training, dev and testing set.  The data will be saved in `corpus.en` and `corpus.ch`, with one sentence in each line.
- **Word segmentation model training**: Run `./tokenizer/tokenize.py`, in which the *sentencepiece.SentencePieceTrainer.Train()* mothed is called to train our word segmentation model. After training, `chn.model`ï¼Œ`chn.vocab`ï¼Œ`eng.model` and `eng.vocab` will be saved in `./tokenizer`.  `.model` is the word segmentation model we need and `.vocab` is the vocabulary.

## Model

We use the open-source code [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) developmented by Harvard.

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- pytorch >= 1.5.1
- sacrebleu >= 1.4.14
- sentencepiece >= 0.1.94

To get the environment settled quickly, run:

```
pip install -r requirements.txt
```

## Usage

Hyperparameters can be modified in `config.py`.

- This code supports MultiGPU training. You should modify `device_id` list in  `config.py` and `os.environ['CUDA_VISIBLE_DEVICES']` in `main.py` to use your own GPUs.

To start training, please run:

```
python main.py
```

The training log is saved in `./experiment/train.log`, and the translation results of testing dataset is in `./experiment/output.txt`.

> Training on 2 GeForce GTX 1080 Ti, 1h/epoch.

## Results

| Model | NoamOpt | LabelSmoothing | Best Dev Bleu | Test Bleu |
| :---: | :-----: | :------------: | :-----------: | :-------: |
|   1   |   No    |       No       |     24.07     |   24.03   |
|   2   |   Yes   |       No       |   **26.08**   | **25.94** |
|   3   |   No    |      Yes       |     23.92     |   23.84   |

## Pretrained Model

You can email me if you need the pretrained model (Model 2 -- The best performance model)ðŸ˜Š. I will send you a google drive download link.

## Beam Search

The testing results of Model 2 with beam search:

| Beam_size |   2   |   3   |   4   |     5     |
| :-------: | :---: | :---: | :---: | :-------: |
| Test Bleu | 26.59 | 26.80 | 26.84 | **26.86** |

## One Sentence Translation

Name the pretrained model or your own trained model with `model.pth` and save it in the path `./experiment`. Run `translate_example` method in `main.py`, and then you can get one sentence translation result.

English Input Sentence for example:

```
The near-term policy remedies are clear: raise the minimum wage to a level that will keep a fully employed worker and his or her family out of poverty, and extend the earned-income tax credit to childless workers.
```

ground truth:

```
è¿‘æœŸçš„æ”¿ç­–å¯¹ç­–å¾ˆæ˜Žç¡®ï¼šæŠŠæœ€ä½Žå·¥èµ„æå‡åˆ°è¶³ä»¥ä¸€ä¸ªå…¨èŒå·¥äººåŠå…¶å®¶åº­å…äºŽè´«å›°çš„æ°´å¹³ï¼Œæ‰©å¤§å¯¹æ— å­å¥³åŠ³åŠ¨è€…çš„å·¥èµ„æ‰€å¾—ç¨Žå‡å…ã€?
```

Translation result with beam size = 3:

```
çŸ­æœŸæ”¿ç­–æ–¹æ¡ˆå¾ˆæ¸…æ¥?æŠŠæœ€ä½Žå·¥èµ„æé«˜åˆ°å……åˆ†å°±ä¸šçš„æ°´å¹?å¹¶æ‰©å¤§å‘æ— è–ªå·¥äººå‘æ”¾æ‰€å¾—çš„ç¨Žæ”¶ä¿¡ç”¨ã€?
```

## Mention

The codes released in this reposity are only tested successfully with **Linux**. If you wanna try it with **Windows**, steps below may be useful to you as mentioned in [issue 2](https://github.com/hemingkx/ChineseNMT/issues/2):

1. **adding utf-8 encoding declaration:**

   in lines 16 and 19 of get_corpus.py:

   ```
   with open(ch_path, "w", encoding="utf-8") as fch:
   with open(en_path, "w", encoding="utf-8") as fen:
   ```

   in line 165 of train.py:

   ```
   with open(config.output_path, "w", encoding="utf-8") as fp:
   ```

2. **using conda command to install sacrebleu if Anoconda is used for building your virtual env:**

   ```
   conda install -c conda-forge sacrebleu
   ```

For any other problems you meet when doing your own project, welcome to issuing or sending emails to me ðŸ˜Š~

