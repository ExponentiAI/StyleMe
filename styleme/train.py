############################
#       main training      #
############################

import train_step_1
import train_step_2
from config import TRAIN_AE_ONLY, TRAIN_GAN_ONLY


if __name__ == "__main__":
    if TRAIN_GAN_ONLY:
        print('train gan only !')
        train_step_1.train()
    else:
        print('train ae first !')
        train_step_1.train()
        if not TRAIN_AE_ONLY:
            train_step_2.train()
