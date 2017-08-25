import codecs
import bitstring
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps
import os

# (1)
IMAGE_SIZE = 30
RECORD_SIZE = 2745
SAMPLE_SIZE = 52796

records = {1: 9056, 2: 10480, 3: 11360, 4: 10480, 5: 11420}

t56s = "0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);'|/STUVWXYZ ,%=\"!"

# (2)
def t56(char):
    return t56s[char]

# (3)
with codecs.open("co59-utf8.txt", "r", "utf-8") as co59f:
    co59t = co59f.read()
co59l = co59t.split()
CO59 = {}
for c in co59l:
    ch = c.split(":")
    co = ch[1].split(",")
    CO59[(int(co[0]), int(co[1]))] = ch[0]

# (4)
def get_char(one_hot_array):
    char = np.argmax(one_hot_array)
    print("{}: \"{}\"".format(char, chr(char)))
    return chr(char)

# (5)
def read_data(directory):
    for i in range(1, 6):
        filename = "{}/ETL2_{}".format(directory, i)
        f = bitstring.ConstBitStream(filename=filename)
        num_of_records = records[i]
        for j in tqdm(range(num_of_records)):
            f.pos = j * 6 * 3660
            r = f.readlist("int:36,uint:6,pad:30,6*uint:6,6*uint:6,pad:24,2*uint:6,pad:180,bytes:2700")
            # print(r[0], t56(r[1]), "".join(map(t56, r[2:8])), "".join(map(t56, r[8:14])), CO59[tuple(r[14:16])])

            image = Image.frombytes("F", (60, 60), r[16], "bit", 6)
            image = image.convert("L")
            enhancer = ImageEnhance.Brightness(image)
            enhanced_image = enhancer.enhance(6)
            small_image = enhanced_image.resize([IMAGE_SIZE, IMAGE_SIZE])
            inverted_image = ImageOps.invert(small_image)

            char_code = ord(CO59[tuple(r[14:16])])
            if not os.path.exists("pic_b/{}".format(char_code)):
                os.makedirs("pic_b/{}".format(char_code))
            image_name = "pic_b/{}/{:d}.png".format(char_code, r[0])
            inverted_image.save(image_name, "PNG")

# (6)
read_data("data")
