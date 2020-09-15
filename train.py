import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import os
from PIL import Image

from keras_retinanet.utils.visualization import draw_box

directory = 'dataset/train/'
train_x_full = np.load(directory + 'train_x.npy')
train_y_prob_full = np.load(directory + 'train_y_prob.npy')
train_y_keys_full = np.load(directory + 'train_y_keys.npy')

converted_data_train = {
    'image': [],
    'x_min': [],
    'y_min': [],
    'x_max': [],
    'y_max': [],
    'class_name': [],
}


if not os.path.exists('fingertips'):
    os.mkdir('fingertips')


fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
for i in range(25090):    
    c = 0
    
    for j in train_y_prob_full[i]:
        if j == 1.0:
            c += 1    
    x = True
    finger_counter = 0
    for k in range(0, len(train_y_keys_full[i]) - 1, 2):
        if int(train_y_keys_full[i][k][0]) != 0.0:
            converted_data_train['image'].append(f'images/{i}.jpg')
            converted_data_train['class_name'].append(fingers[finger_counter])
            converted_data_train['x_min'].append(int(train_y_keys_full[i][k][0]))
            converted_data_train['x_max'].append(int(train_y_keys_full[i][k][0]) + 1)
            converted_data_train['y_min'].append(int(train_y_keys_full[i][k + 1][0]))
            converted_data_train['y_max'].append(int(train_y_keys_full[i][k + 1][0]) + 1)  
        finger_counter += 1


train_df = pd.DataFrame(converted_data_train)
train_df.head()


def show_image_with_boxes(df):
    filepath = df.sample()['image'].values[0]

    df2 = df[df['image'] == filepath]
    im = np.array(Image.open(filepath))

    im = im[:, :, :3]

    for idx, row in df2.iterrows():
        box = [
          row['x_min'],
          row['y_min'],
          row['x_max'],
          row['y_max'],
        ]
        print(box)
        if box != [0, 0, 1, 1]:
            draw_box(im, box, color=(255, 0, 0))

    plt.axis('off')
    plt.imshow(im)
    plt.show()


train_df.to_csv('annotations.csv', index=False, header=False)


classes = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
with open('classes.csv', 'w+') as f:
    for i, class_name in enumerate(classes):
        f.write(f'{class_name},{i}\n')

if not os.path.exists('snapshots'):
    os.mkdir('snapshots')

PRETRAINED_MODEL = 'snapshots/_pretrained_model.h5'

URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)

print('Downloaded pretrained model to ' + PRETRAINED_MODEL)

os.system(f'python keras-retinanet/keras_retinanet/bin/train.py --freeze-backbone   --random-transform  \
 --weights {PRETRAINED_MODEL}   --batch-size 2   --steps 500   --epochs 15   csv annotations.csv classes.csv')
