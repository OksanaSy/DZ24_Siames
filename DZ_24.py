import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import transformers

transformers.logging.set_verbosity_error()

df = pd.read_csv("train.csv")
def create_pairs(df, num_pairs=15000):
    same_author_pairs = []
    different_author_pairs = []

    grouped = df.groupby('author')
    for author, group in grouped:
        texts = group['text'].values
        same_author_pairs += list(itertools.combinations(texts, 2))
        if len(same_author_pairs) >= num_pairs // 2:
            break

    other_authors = list(set(df['author'].unique()) - {author})
    for text in df[df['author'] == author]['text'].sample(num_pairs // 2):
        different_author = np.random.choice(other_authors)
        different_text = df[df['author'] == different_author]['text'].sample(1).values[0]
        different_author_pairs.append((text, different_text))

    pairs = same_author_pairs[:num_pairs // 2] + different_author_pairs
    labels = [1] * (num_pairs // 2) + [0] * (num_pairs // 2)

    return pairs, labels

pairs, labels = create_pairs(df)

pairs_train, pairs_test, y_train, y_test = train_test_split(
    pairs, labels, test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_pairs(tokenizer, pairs):
    texts1 = [p[0] for p in pairs]
    texts2 = [p[1] for p in pairs]
    return (
        tokenizer(texts1, padding=True, truncation=True, max_length=128, return_tensors='tf'),
        tokenizer(texts2, padding=True, truncation=True, max_length=128, return_tensors='tf')
    )

tokenized_pairs_train1, tokenized_pairs_train2 = tokenize_pairs(tokenizer, pairs_train)
tokenized_pairs_test1, tokenized_pairs_test2 = tokenize_pairs(tokenizer, pairs_test)

class MeanPoolingLayer(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

class BertLayer(Layer):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

class L1DistanceLayer(Layer):
    def call(self, inputs):
        pooled_output1, pooled_output2 = inputs
        return tf.abs(pooled_output1 - pooled_output2)

def create_siamese_model():
    # Вхідні дані для двох текстів у парі
    input1_ids = Input(shape=(128,), dtype=tf.int32, name="input1_ids")
    input1_mask = Input(shape=(128,), dtype=tf.int32, name="input1_mask")
    input2_ids = Input(shape=(128,), dtype=tf.int32, name="input2_ids")
    input2_mask = Input(shape=(128,), dtype=tf.int32, name="input2_mask")

    # Спільний шар BERT для обох вхідних текстів
    bert_layer = BertLayer()
    output1 = bert_layer([input1_ids, input1_mask])
    output2 = bert_layer([input2_ids, input2_mask])

    # Пулінг шар для зведення до одного вектора
    pooled_output1 = MeanPoolingLayer()(output1)
    pooled_output2 = MeanPoolingLayer()(output2)

    # Обчислення L1 відстані між двома векторами
    l1_distance = L1DistanceLayer()([pooled_output1, pooled_output2])
    output = Dense(1, activation="sigmoid")(l1_distance)

    model = Model(inputs=[input1_ids, input1_mask, input2_ids, input2_mask], outputs=output)
    model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["AUC"])
    return model

siamese_model = create_siamese_model()
siamese_model.fit(
    [tokenized_pairs_train1["input_ids"], tokenized_pairs_train1["attention_mask"],
     tokenized_pairs_train2["input_ids"], tokenized_pairs_train2["attention_mask"]],
    np.array(y_train),
    validation_split=0.2,
    epochs=10,
    batch_size=16
)

test_results = siamese_model.evaluate(
    [tokenized_pairs_test1["input_ids"], tokenized_pairs_test1["attention_mask"],
     tokenized_pairs_test2["input_ids"], tokenized_pairs_test2["attention_mask"]],
    np.array(y_test)
)
print(f"Test Loss: {test_results[0]}, Test AUC: {test_results[1]}")

"""
Epoch 1/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 734s 1s/step - AUC: 0.3258 - loss: 0.7166 - val_AUC: 0.3932 - val_loss: 0.7089
Epoch 2/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 722s 1s/step - AUC: 0.4352 - loss: 0.7036 - val_AUC: 0.5064 - val_loss: 0.6956
Epoch 3/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 742s 1s/step - AUC: 0.5564 - loss: 0.6898 - val_AUC: 0.6172 - val_loss: 0.6826
Epoch 4/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 740s 1s/step - AUC: 0.6570 - loss: 0.6784 - val_AUC: 0.7159 - val_loss: 0.6700
Epoch 5/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 1194s 2s/step - AUC: 0.7573 - loss: 0.6643 - val_AUC: 0.7969 - val_loss: 0.6576
Epoch 6/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 706s 1s/step - AUC: 0.8305 - loss: 0.6521 - val_AUC: 0.8571 - val_loss: 0.6455
Epoch 7/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 750s 1s/step - AUC: 0.8862 - loss: 0.6399 - val_AUC: 0.9002 - val_loss: 0.6337
Epoch 8/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 730s 1s/step - AUC: 0.9213 - loss: 0.6290 - val_AUC: 0.9310 - val_loss: 0.6222
Epoch 9/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 710s 1s/step - AUC: 0.9467 - loss: 0.6159 - val_AUC: 0.9506 - val_loss: 0.6108
Epoch 10/10
600/600 ━━━━━━━━━━━━━━━━━━━━ 743s 1s/step - AUC: 0.9605 - loss: 0.6060 - val_AUC: 0.9642 - val_loss: 0.5998
94/94 ━━━━━━━━━━━━━━━━━━━━ 198s 2s/step - AUC: 0.9676 - loss: 0.5993
Test Loss: 0.597958505153656, Test AUC: 0.9703071117401123

Process finished with exit code 0

"""
