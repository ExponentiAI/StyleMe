#################################
#       training parameter      #
#################################

DATALOADER_WORKERS = 2
NBR_CLS = 50

EPOCH_GAN = 100
ITERATION_GAN = 2000

SAVE_IMAGE_INTERVAL = 100
SAVE_MODEL_INTERVAL = 200
LOG_INTERVAL = 200
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

# PRETRAINED_AE_PATH = './checkpoint/models/AE_20000.pth'
PRETRAINED_AE_PATH = None

# GAN_CKECKPOINT = './checkpoint/models/9.pth'
GAN_CKECKPOINT = None

TRAIN_AE_ONLY = False
TRAIN_GAN_ONLY = False

data_root_colorful = './train_data/rgb/'
data_root_sketch = './train_data/sketch_styleme/'
# data_root_sketch = './train_data/sketchgen_wo_cam/'
# data_root_sketch = './train_data/sketchgen_wo_adalin/'
# data_root_sketch = './train_data/sketchgen_wo_camada/'
