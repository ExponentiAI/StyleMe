# -*- coding:utf8 -*-

import os


class BatchRename():
    def __init__(self):
        # self.path = './sketch_styletransfer/image/rgb/rgb/'
        # self.path = './sketch_styletransfer/train_data/sketch/data/'
        self.path = './train_data_3000/art_sketch/'

    def rename(self):
        file = os.listdir(self.path)
        total_num = len(file)
        i = 0
        for item in range(total_num):
            src = os.path.join(os.path.abspath(self.path), str(item) + '.jpg')
            dst = os.path.join(os.path.abspath(self.path), str(i).zfill(4) + '.jpg')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
