from encoder import train as train_ae
from selfgan import train as train_gan


if __name__ == "__main__":
    
    from config import TRAIN_AE_ONLY, TRAIN_GAN_ONLY
    if TRAIN_GAN_ONLY:
        print('train gan only !')
        train_gan()
    else:
        print('train ae first !')
        train_ae()
        if not TRAIN_AE_ONLY:
            train_gan()
