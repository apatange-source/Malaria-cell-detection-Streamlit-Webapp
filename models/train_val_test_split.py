import numpy as np
import shutil
import os

def split(root_dir, ratio=[0.8, 0.1, 0.1]):

    [posCls, negCls] = [name for name in os.listdir(root_dir)]
    posCls = '/' + posCls
    negCls = '/' + negCls
    if not os.path.exists(root_dir + '/train' + posCls):
        os.makedirs(root_dir + '/train' + posCls)
    if not os.path.exists(root_dir + '/train' + negCls):
        os.makedirs(root_dir + '/train' + negCls)
    if not os.path.exists(root_dir + '/val' + posCls):
        os.makedirs(root_dir + '/val' + posCls)
    if not os.path.exists(root_dir + '/val' + negCls):
        os.makedirs(root_dir + '/val' + negCls)
    if not os.path.exists(root_dir + '/test' + posCls):
        os.makedirs(root_dir + '/test' + posCls)
    if not os.path.exists(root_dir + '/test' + negCls):
        os.makedirs(root_dir + '/test' + negCls)

    currentCls = posCls
    src = root_dir + currentCls

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * ratio[0]),
                                                               int(len(allFileNames) * (ratio[0] + ratio[1]))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + '/train' + currentCls)

    for name in val_FileNames:
        shutil.copy(name, root_dir + '/val' + currentCls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + '/test' + currentCls)

    currentCls = negCls
    src = root_dir + currentCls

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * ratio[0]),
                                                               int(len(allFileNames) * (ratio[0] + ratio[1]))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir + '/train' + currentCls)

    for name in val_FileNames:
        shutil.copy(name, root_dir + '/val' + currentCls)

    for name in test_FileNames:
        shutil.copy(name, root_dir + '/test' + currentCls)

