
DATALOADER_WORKERS = 6
NBR_CLS = 50

EPOCH_GAN = 10
ITERATION_GAN = 2000

SAVE_IMAGE_INTERVAL = 100
SAVE_MODEL_INTERVAL = 200
LOG_INTERVAL = 100
FID_INTERVAL = 100
FID_BATCH_NBR = 100

ITERATION_AE = 20000

CHANNEL = 32
MULTI_GPU = False

IM_SIZE_GAN = 256
BATCH_SIZE_GAN = 8

IM_SIZE_AE = 256
BATCH_SIZE_AE = 8

SAVE_FOLDER = './checkpoint/'
# SAVE_FOLDER = './train_data_3000/checkpoint/'

PRETRAINED_AE_PATH = './checkpoint/train_results/AE_20000.pth'
# PRETRAINED_AE_PATH = './train_data_3000/checkpoint/train_results/AE_1200.pth'
# PRETRAINED_AE_PATH = None

GAN_CKECKPOINT = None

TRAIN_AE_ONLY = False
TRAIN_GAN_ONLY = False

data_root_colorful = './train_data/art_rgb/'
data_root_sketch = './train_data/art_sketch/sketch/'

# data_root_colorful = './train_data_3000/art_rgb/'
# data_root_sketch = './train_data_3000/art_sketch/'
