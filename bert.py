from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import argparse
import zipfile
import emoji as emoji
import sklearn.metrics
import pandas as pd
from keras import Model, Sequential, Input
from keras.layers import Dense, Lambda
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import numpy as np
from bs4 import BeautifulSoup
from keras_bert import Tokenizer

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

def emoj(tweet):
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", " ")
    tweet = ' '.join(tweet.split())
    return tweet

config_path = '/Users/weigoxu/Downloads/bert_base/bert_config.json'
checkpoint_path = '/Users/weigoxu/Downloads/bert_base/bert_model.ckpt'
dict_path = '/Users/weigoxu/Downloads/bert_base/vocab.txt'

###################################################################################################

def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:

    # form train data and dev data
    train_in = train_data["Tweet"]
    dev_in = dev_data["Tweet"]

    # take off @users
    train_in = train_in.apply(lambda x: ' '.join([w for w in x.split() if not w.startswith('@')]))
    dev_in = dev_in.apply(lambda x: ' '.join([w for w in x.split() if not w.startswith('@')]))
    #take off #
    train_in  = [w.replace('#', '') for w in train_in]
    dev_in  = [w.replace('#', '') for w in dev_in]
    # transform emoji
    train_in = [emoj( BeautifulSoup(tt).get_text()) for tt in train_in]
    dev_in = [emoj( BeautifulSoup(tt).get_text()) for tt in dev_in]

    train_in = np.array(train_in)
    dev_in = np.array(dev_in)
    train_out = np.array(train_data[emotions])
    dev_out = np.array(dev_data[emotions])

    # load bert model
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
            l.trainable = True

    # create dict from bert
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)
    maxlen = 50

    #encode
    #get indices at [0], segments is at [1]
    x_train_word_indices = [tokenizer.encode(tt)[0] for tt in train_in]
    x_dev_word_indices = [tokenizer.encode(tt)[0] for tt in dev_in]

    x_train_padded_seqs = pad_sequences(x_train_word_indices, maxlen=maxlen, padding='post')
    x_dev_padded_seqs = pad_sequences(x_dev_word_indices, maxlen=maxlen, padding='post')

    x_train_word_segments = np.zeros(x_train_padded_seqs.shape)
    x_dev_word_segments = np.zeros(x_dev_padded_seqs.shape)

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in,x2_in])
    #cls at 0
    print(x.shape)
    x = Lambda(lambda x: x[:, 0])(x)
    print(x.shape)
    p = Dense(11, activation='sigmoid')(x)

    print("okman",p.shape)
    model = Model([x1_in,x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  #small lr
        metrics=['accuracy']
    )
    model.summary()

    kwargs = {}
    kwargs.update(x=[x_train_padded_seqs,x_train_word_segments], y=train_out,
                  validation_data=([x_dev_padded_seqs,x_dev_word_segments], dev_out),epochs=3,shuffle=True)
    print(train_out.shape)
    model.fit(**kwargs)

    prediction = model.predict([x_dev_padded_seqs,x_dev_word_segments])
    print(prediction.shape)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0

    dev_predictions = dev_data.copy()
    dev_predictions[emotions] = prediction.astype(np.int)

    return dev_predictions

if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="2018-E-c-En-dev.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_data, test_data)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))