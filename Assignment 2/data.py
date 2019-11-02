import os, cv2, random
import numpy as np

ROWS = 50
COLS = 50

def get_train(truncate=False):
    TRAIN_DIR = 'data/train/'
    
    train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
    train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
    train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
    
    # only use portion of dataset
    if truncate:
        train_images = train_dogs[:1250] + train_cats[:1250]
    else:
        train_images = train_dogs + train_cats
    random.shuffle(train_images)
    
    train = prep_data(train_images)
    
    print("Train shape: {}".format(train.shape))
    
    labels = []
    for i in train_images:
        if 'dog' in i:
            labels.append(1)
        else:
            labels.append(0)
    
    
    # TEST_DIR = 'data/test/'
    # test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    # test_images =  test_images[:25]
    # test = prep_data(test_images)
    # print("Test shape: {}".format(test.shape))
    
    return train/255, np.array(labels)


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        # if i%500 == 0: print('Processed {} of {}'.format(i, count))
    
    return data.reshape((data.shape[0], 1, data.shape[1]*data.shape[2]))


if __name__ == '__main__':
    get_train()
