import sys
import os
from keras.optimizers import Adam
import zipfile
import emoji as emoji
import sklearn.metrics
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
from keras_xlnet import PretrainedList, get_pretrained_paths


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

EPOCH = 5
BATCH_SIZE = 128
SEQ_LEN = 40



def emoj(tweet):
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", " ")
    tweet = ' '.join(tweet.split())
    return tweet



import os
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
paths = get_pretrained_paths(PretrainedList.en_cased_base)
tokenizer = Tokenizer(paths.vocab)

model = load_trained_model_from_checkpoint(
    config_path=paths.config,
    checkpoint_path=paths.model,
    batch_size=BATCH_SIZE,
    memory_len=0,
    target_len=SEQ_LEN,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)

last = Extract(index=-1, name='Extract')(model.output)
print(last.shape)
dense = keras.layers.Dense(units=768, activation='relu', name='Dense')(last)
dropout = keras.layers.Dropout(rate=0.15, name='Dropout')(dense)
output = keras.layers.Dense(units=11, activation='sigmoid', name='Sigmoid')(dropout)
print(output.shape)
model = keras.models.Model(inputs=model.inputs, outputs=output)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(3e-5),  #small lr
    metrics=['accuracy']
)
model.summary()
###################################################################################################
class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index):
        s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        return [item[s] for item in self.x], self.y[s]

def generate_sequence(tweets,labels):
    tokens, classes = [], []
    for tweet in tweets:
        encoded = tokenizer.encode(tweet)[:SEQ_LEN - 1]
        encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
        tokens.append(encoded)
    for label in labels:
        classes.append(label.astype(np.int))

    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)


def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:
    # form train data and dev data
    train_in = train_data["Tweet"]
    dev_in = dev_data["Tweet"]

    # take off @users
    train_in = train_in.apply(lambda x: ' '.join([w for w in x.split() if not w.startswith('@')]))
    dev_in = dev_in.apply(lambda x: ' '.join([w for w in x.split() if not w.startswith('@')]))
    # take off #
    train_in  = [w.replace('#', '') for w in train_in]
    dev_in  = [w.replace('#', '') for w in dev_in]
    # transform emoji
    train_in = [emoj(BeautifulSoup(tt).get_text()) for tt in train_in]
    dev_in = [emoj(BeautifulSoup(tt).get_text()) for tt in dev_in]

    train_in, dev_in = np.array(train_in), np.array(dev_in)
    train_out, dev_out = np.array(train_data[emotions]), np.array(dev_data[emotions])

    train_seq = generate_sequence(train_in,train_out)
    dev_seq = generate_sequence(dev_in,dev_out)

    model.fit_generator(
        generator=train_seq,
        validation_data=dev_seq,
        epochs=EPOCH,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
    )

    prediction = model.predict_generator(dev_seq, verbose=True)
    print(prediction.shape)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0

    dev_predictions = dev_data.copy()
    dev_predictions[emotions] =  prediction.astype(np.int)

    return dev_predictions

###################################################################################################
if __name__ == "__main__":

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv("./2018-E-c-En-train.txt", **read_csv_kwargs)
    test_data = pd.read_csv("./2018-E-c-En-dev.txt", **read_csv_kwargs)

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_data, test_data)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))
