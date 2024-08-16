
import os
from datetime import datetime


CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_TRAIN_STD = (0.2023, 0.1994, 0.2010)


#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 250
MILESTONES = [120, 160, 200]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








