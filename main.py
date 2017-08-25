# 参考資料
# ダービッドさんのコード
# 「TensorFlowで学ぶディープラーニング入門、５章」https://github.com/enakai00/jupyter_tfbook/blob/master/Chapter05/MNIST%20double%20layer%20CNN%20classification.ipynb

import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
# from layers import*

RUN = "_new_eval"
DATA_DIR = "obj_360"
LOG_DIR = "log_0608"
SAVE_LOC = "/".join((LOG_DIR, "model"))
MAX_STEPS = 10000
BATCH_SIZE = 200
IMAGE_SIZE = 28
dropout_rate = 0.2

# global_train_step = 0
# global_test_step = 0

# ランダムのシードを設定
np.random.seed(201705)

# バッチを作るときに使う関数
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation, :]
    return shuffled_dataset, shuffled_labels
def create_batch(dataset, labels, BATCH_SIZE):
    shuffled_dataset, shuffled_labels = randomize(dataset, labels)
    batch_dataset = shuffled_dataset[0:BATCH_SIZE]
    batch_labels = shuffled_labels[0:BATCH_SIZE]
    return batch_dataset, batch_labels



def load_obj(name):
    with open("{}/{}.pkl".format(DATA_DIR, name), "rb") as f:
        return pickle.load(f)

def get_code_dictionary():
    return load_obj("code_dic")

def get_unicode_dictionary():
    return load_obj("unicode_dic")

# 重み関数を作る関数
def weighted_variable(shape, w_name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアスを作る関数
def bias_variable(shape, b_name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=b_name)

# （レイヤーを作る関数の中で使用）
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        tf.summary.histogram("histogram", var)


def create_nnLayer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            weights = weighted_variable([input_dim, output_dim], w_name=layer_name+"_weights")
            variable_summaries(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_dim], b_name=layer_name+"_bias")
            variable_summaries(biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram("preactivate", preactivate)
            activations = act(preactivate, name="activation")
            output = activations
    return (weights, biases, output)


def create_cnnLayer(input_tensor, patch, input_channels, output_channels, layer_name, pool=False, dropout=False, dropout_rate=None, dropout_mode=None):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
                weights = weighted_variable([patch, patch, input_channels, output_channels], w_name=layer_name+"_weights")
                variable_summaries(weights)
        with tf.name_scope("biases"):
            biases = bias_variable([output_channels], b_name=layer_name+"_bias")
            variable_summaries(biases)
        conv = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.relu(conv + biases)
        if not pool:
            if not dropout:
                output = conv
            else:
                dropped = tf.layers.dropout(inputs=conv, rate=dropout_rate, training=dropout_mode, name="dropout")
                output = dropped
        else:
            pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if not dropout:
                output = pool
            else:
                dropped = tf.layers.dropout(inputs=pool, rate=dropout_rate, training=dropout_mode, name="dropout")
                output = dropped
    return (weights, biases, output)




# etl2input.pyで作ったデータとラベルの読み込み
f = open("{}/characters_dataset".format(DATA_DIR), "rb")
x_train = np.load(f)
y_train = np.load(f)
x_test = np.load(f)
y_test = np.load(f)
label_names = np.load(f)
f.close()

NUMBER_OF_CLASSES = y_train.shape[1]


# ノードやレイヤの定義

# インプットの入れ物を定義
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE], name="image_data")
    y = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES], name="label")
    x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1], name="reshaped_input")

# CNN層でドロップアウトをするかどうかを決めるスイッチ。
training_mode = tf.placeholder(tf.bool, name="training_mode")

# レイヤーの定義 [defined in layers.py]
# １段目の畳み込みレイヤーの定義
cnn1_output_dim = 32
cnn1_weights, cnn1_biases, cnn1_output = create_cnnLayer(input_tensor=x_image, patch=7, input_channels=1, output_channels=cnn1_output_dim, layer_name='cnn1', pool=True, dropout=True, dropout_rate=dropout_rate, dropout_mode=training_mode)

# 2段目の畳み込みレイヤーの定義
cnn2_output_dim = 32
cnn2_weights, cnn2_biases, cnn2_output = create_cnnLayer(input_tensor=cnn1_output, patch=5, input_channels=cnn1_output_dim, output_channels=cnn2_output_dim, layer_name='cnn2', pool=False, dropout=True, dropout_rate=dropout_rate, dropout_mode=training_mode)

# 3段目の畳み込みレイヤー
cnn3_output_dim = 64
cnn3_weights, cnn3_biases, cnn3_output = create_cnnLayer(input_tensor=cnn2_output, patch=5, input_channels=cnn2_output_dim, output_channels=cnn3_output_dim, layer_name='cnn3', pool=True, dropout=True, dropout_rate=dropout_rate, dropout_mode=training_mode)
# 4段目の畳み込みレイヤーの定義
cnn4_output_dim = 64
cnn4_weights, cnn4_biases, cnn4_output = create_cnnLayer(input_tensor=cnn3_output, patch=3, input_channels=cnn3_output_dim, output_channels=cnn4_output_dim, layer_name='cnn4', pool=False)


# 全結合層、ドロップアウト層、ソフトマックス関数の定義
num_units1 = (IMAGE_SIZE//4)*(IMAGE_SIZE//4)*cnn4_output_dim #IMAGE_SIZEを変えるときはの計算に注意
num_units2 = 1024
nn1_input = tf.reshape(cnn4_output, [-1, num_units1])

# 全結合層
nn1_weights, nn1_biases, nn1_output = create_nnLayer(input_tensor=nn1_input, input_dim=num_units1, output_dim=num_units2, layer_name='nn1', act=tf.nn.relu)

# ドロップアウト層
with tf.name_scope('dropout_last'):
    nn1_drop = tf.nn.dropout(nn1_output, 1-dropout_rate, name="nn1_drop")

# ソフトマックス層
softmax_weights, softmax_biases, softmax_output = create_nnLayer(input_tensor=nn1_drop, input_dim=num_units2, output_dim=NUMBER_OF_CLASSES, layer_name='softmax', act=tf.nn.softmax)
p = softmax_output

# 誤差関数 loss、トレーニングアルゴリズム train_step、正解率 accuracy を定義します。
with tf.name_scope("cross_entropy"):
    loss = -tf.reduce_sum(y * tf.log(tf.clip_by_value(p,1e-10,1.0)))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss) #トレーニング本体

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1)) #正解不正解を表すboolean配列の作成。　argmax(p,1)はテンソルpのそれぞれの第1軸において一番大きいものを選ぶ関数。結果的に第1軸が消える。
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast: booleanの配列をfloat32の配列へ、つまり0と1の配列に変換。それをもとに正解率をだす。
    tf.summary.scalar("accuracy", accuracy)

# SummaryWriterのインスタンスを取得 (参考： http://qiita.com/sergeant-wizard/items/af7a3bd90e786ec982d2)
# summary_writer = tf.train.SummaryWriter(LOG_DIR)
merged = tf.summary.merge_all()

# セッションを用意して、Variable を初期化します。
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
test_writer = tf.summary.FileWriter(LOG_DIR + "/test" + RUN, graph=sess.graph)

A = sess.run(cnn1_weights)
print(A[:, :, 0, 0])

i = 0
for _ in tqdm(range(MAX_STEPS)):
    i += 1
    batch_xs, batch_ys = create_batch(x_train, y_train, BATCH_SIZE=BATCH_SIZE)
    sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, training_mode:True })
    if i % 100 == 0:
        loss_vals, acc_vals = [], []
        A = sess.run(cnn1_weights)
        print(A[:, :, 0, 0])
        for c in range(4):
            start = int(len(y_test) / 4 * c) # ここでテストを４回に分けて行っている。メモリの使用量を減らす工夫であり、本質にかかわらない。
            end = int(len(y_test) / 4 * (c+1))
            loss_val, acc_val, summary = sess.run([loss, accuracy, merged], feed_dict={x:x_test[start:end], y:y_test[start:end], training_mode:False})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        saver.save(sess, SAVE_LOC, global_step=i)
        test_writer.add_summary(summary)
