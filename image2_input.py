import os
from scipy import misc
import numpy as np
from tqdm import tqdm
import pickle


INPUT_PATH = "pic_b"
OUTPUT_PATH = "obj_360"
SHUFFLE_VALUE = 90


def save_obj(obj, name):
    with open("{}/{}.pkl".format(OUTPUT_PATH, name), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation, :]
    return shuffled_dataset, shuffled_labels



folder_names = []
folder_files = []
examples = []
for name in sorted(os.listdir(INPUT_PATH)): #INPUT_PATHにあるフォルダ一つ一つに同じ操作
    if int(name) < 12354:
        continue
    if int(name) > 12435:
        break
    folder_names.append(name) #フォルダ名をつなげてフォルダ名による１次元配列folder_namesを作成
    files = sorted(os.listdir(os.path.join(INPUT_PATH, name))) #フォルダの中のファイル名の１次元配列filesを作成
    folder_files.append(files) #１次元配列であるfilesをforごとにつなげて２次元配列folder_filesを作成
    examples.append(len(files))
# 例）
# folder_names, examples, folder_files: files:
# name1       , 34     ,               file(1,1), file(1,2), file(1,3),...
# name2
# name3
# name4
# .....

number_of_classes = len(folder_names)
x = []
y = []

unicode_dic = {}
code_dic = {}

# code_index = 0

print("Reading files from {} folders".format(number_of_classes))
for i in tqdm(range(number_of_classes)):
    y_each = np.zeros(number_of_classes)
    y_each[i] = 1
    for j in range(examples[i]):
        img_lctn = os.path.join(INPUT_PATH, folder_names[i], folder_files[i][j])
        image = misc.imread(img_lctn, mode="L")
        if image.shape[0] != 28:
            image = image[1:29, 1:29]
        x.append(image)
        y.append(y_each)
    unicode_dic[int(folder_names[i])] = i
    code_dic[i] = int(folder_names[i])

x = np.float32(x) / 255.0
y = np.int32(y)
print("x: {}, y: {}".format(x.shape, y.shape))

label_names = folder_names

x = np.concatenate([x[i::SHUFFLE_VALUE] for i in range(SHUFFLE_VALUE)])
y = np.concatenate([y[i::SHUFFLE_VALUE] for i in range(SHUFFLE_VALUE)])

dataset_size = x.shape[0]

x, y = randomize(x, y)
x_train = x[:(dataset_size * 8) // 10]
y_train = y[:(dataset_size * 8) // 10]

x_test = x[(dataset_size * 8) // 10:]
y_test = y[(dataset_size * 8) // 10:]


print("x: {}".format(x.shape))
print("y: {}".format(y.shape))
print("x_train: {}".format(x_train.shape))
print("y_train: {}".format(y_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_test: {}".format(y_test.shape))
print("label_names: {}".format(str(len(label_names))))

save_obj(code_dic, "code_dic")
save_obj(unicode_dic, "unicode_dic")

f = open("{}/characters_dataset".format(OUTPUT_PATH), "wb")
np.save(f, x_train)
np.save(f, y_train)
np.save(f, x_test)
np.save(f, y_test)
np.save(f, label_names)
f.close()
