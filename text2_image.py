import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from degrade_image import *

CODE_FILE = "codes.txt"
TEST_PATH = "pic_b"

fonts = {"gothics28": ImageFont.truetype("msgothic.ttc", 28),
         "minchos28": ImageFont.truetype("msmincho.ttc", 28),
         "hgrmbs28": ImageFont.truetype("HGRMB.TTC", 28),
         "hgrgms28": ImageFont.truetype("HGRGM.TTC", 28),
         "hgrsmps28": ImageFont.truetype("HGRSMP.TTF", 28),
         "hgrge28": ImageFont.truetype("HGRGE.TTC", 28),
         "hgrkk28": ImageFont.truetype("HGRKK.TTC", 28),
         "yugoth28": ImageFont.truetype("YuGothR.ttc", 28),
         "gothics27": ImageFont.truetype("msgothic.ttc", 27),
         "minchos27": ImageFont.truetype("msmincho.ttc", 27),
         "hgrmbs27": ImageFont.truetype("HGRMB.TTC", 27),
         "hgrgms27": ImageFont.truetype("HGRGM.TTC", 27),
         "hgrsmps27": ImageFont.truetype("HGRSMP.TTF", 27),
         "hgrge27": ImageFont.truetype("HGRGE.TTC", 27),
         "hgrkk27": ImageFont.truetype("HGRKK.TTC", 27),
         "yugoth27": ImageFont.truetype("YuGothR.ttc", 27),
         "gothics26": ImageFont.truetype("msgothic.ttc", 26),
         "minchos26": ImageFont.truetype("msmincho.ttc", 26),
         "hgrmbs26": ImageFont.truetype("HGRMB.TTC", 26),
         "hgrgms26": ImageFont.truetype("HGRGM.TTC", 26),
         "hgrsmps26": ImageFont.truetype("HGRSMP.TTF", 26),
         "hgrge26": ImageFont.truetype("HGRGE.TTC", 26),
         "hgrkk26": ImageFont.truetype("HGRKK.TTC", 26),
         "yugoth26": ImageFont.truetype("YuGothR.ttc", 26)
         }

rotation = [-3, 3]

position = [-1, 1]

# filter_matrix = np.zeros([28, 28])
# for i in range(0,28):
#      for j in range(0,28):
#             if (i-14)*(i-14) + (j-14)*(j-14) < 9:
#                 filter_matrix[i][j] = 0
#             elif (i-14)*(i-14) + (j-14)*(j-14) > 36:
#                 filter_matrix[i][j] = 0
#             else:
#                 filter_matrix[i][j] = 1

file = open(CODE_FILE, "r")
for name in tqdm(file.readlines()):
    name = str.strip(name)
    if not os.path.exists("{}/{}".format(TEST_PATH, name)):
        os.makedirs("{}/{}".format(TEST_PATH, name))
    for font in fonts:
        if font.endswith("26"):
            offset = 2
        elif font.endswith("27"):
            offset = 1
        else:
            offset = 0
        for pos in position:
            # img = Image.new("L", (28, 28), 255)
            # draw = ImageDraw.Draw(img)
            # draw.text((0 + offset + pos, 0 + offset + pos), chr(int(name)), 0, font=fonts[font])
            # img.save("{}/{}/{}_{}.png".format(TEST_PATH, name, font, pos))
            for rot in rotation:
                deg_img_array = degrade_img(name=name, fnt=fonts[font], pos=pos, offset=offset, rot=rot)
                # fdeg_img_array = np.fft.fft2(deg_img_array)
                # fscrb_img_array = fdeg_img_array * filter_matrix
                # scrb_img_array = np.fft.ifft2(fscrb_img_array)
                # scrb_img_array = scrb_img_array.real
                deg_img = Image.fromarray(deg_img_array)
                deg_img.save("{}/{}/{}_{}_{}.png".format(TEST_PATH, name, font, pos, rot))
                # deg_img = Image.fromarray(scrb_img_array)
                # deg_img.save("{}/{}/{}_{}_{}_s.png".format(TEST_PATH, name, font, pos, rot))
file.close()
