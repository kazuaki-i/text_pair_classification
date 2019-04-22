# text_pair_classification

This is an program of text **PAIR** classification using typical neural networks inspired by [Neural Networks for Text Classification](https://github.com/chainer/chainer/tree/master/examples/text_classification), which is one of Chainer official example. 

This code can switch choices below:

- LSTM
- CNN + MLP
- BoW + MLP
- GRU **<- ADDITIONAL!!**


# How to RUN

To train model:

```
python train_text_classifier.py -g 0 --datase traindata --model gru
```

This trainer is required training dataset UNLIKE the original one
The training dataset format is below:
```
input1(words separated space) \t input2(words separated space) \t label
```

This program has additional options without the original example:
- dataset: a input dataset (required)
- size: development dataset size
- early-stop: apply early-stop method to training
- same-network: share same network between input1 and input2
- save-init: save init model

The output directory result contains:

- best_model.npz: a model snapshot, which won the best accuracy for validation data during training
- vocab.json: model's vocabulary dictionary as a json file
- args.json: model's setup as a json file, which also contains paths of the model and vocabulary

To apply the saved model to your sentences, feed the sentences through stdin:

```
cat sentences_to_be_classifed.txt | python run_text_pair_classifier.py -g 0 --model-setup result/args.json
```
The sentences_to_be_classifed.txt format is bellow:
```
input1(words separated space) \t input2(words separated space)
```

The classification result is given by stdout.

---

# text_pair_classification について

このプログラムはニューラルネットワークを使用した、「入力２文」のテキスト分類モデルです。

Chainer 公式 example の [Neural Networks for Text Classification](https://github.com/chainer/chainer/tree/master/examples/text_classification)を改造したものになります。

このプログラムは以下の４つのネットワークを採用しています。

- LSTM
- CNN + MLP
- BoW + MLP
- GRU **<- ADDITIONAL!!**

# How to RUN

## 学習

```
python train_text_classifier.py -g 0 --datase traindata --model gru
```

Chainer の exampleとは異なり、以下のフォーマットの入力ファイルの指定が必須です。各自で用意してください

```
文１(スペース区切り) \t 文２(スペース区切り) \t ラベル
```

このプログラムには Chainer example のオプションに加え、以下のオプションが追加されています。

- dataset: a input dataset (required)
- size: development dataset size
- early-stop: apply early-stop method to training
- same-network: share same network between input1 and input2
- save-init: save init model

出力は result (デフォルト) に以下のデータが保存されます

- best_model.npz: a model snapshot, which won the best accuracy for validation data during training
- vocab.json: model's vocabulary dictionary as a json file
- args.json: model's setup as a json file, which also contains paths of the model and vocabulary

## 予測
学習したモデルを使用して、ラベル予測ができます。
```
cat sentences_to_be_classifed.txt | python run_text_pair_classifier.py -g 0 --model-setup result/args.json
```
sentences_to_be_classifed.txt のフォーマットは以下のとおりです
```
文１(スペース区切り) \t 文２(スペース区切り)
```

予測結果は標準出力で出力されます。