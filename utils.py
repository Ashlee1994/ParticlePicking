# encoding: utf-8

import os,random
import pandas as pd
import numpy as np
import mrcfile
import cv2 as cv
from PIL import Image

def mapstd(mrcData,dim_x,dim_y):
    avg = mrcData.mean()
    stddev = np.std(mrcData,ddof=1)
    minval = mrcData.min()
    maxval = mrcData.max()
    data = mrcData.copy()
    sigma_contrast = 0.3
    if minval == maxval or sigma_contrast > 0.:
        minval = avg - sigma_contrast * stddev
        maxval = avg + sigma_contrast * stddev
    if sigma_contrast > 0 or minval != maxval:
        data[data < minval] = minval
        data[data > maxval] = maxval
    if data.max == 0 and data.min == 0:
        normalize = 0.0
    else:
        data = ( data - data.min())/(data.max() - data.min())
    # 19S data
    data = data[::-1]
    data = data.reshape(dim_y,dim_x)
    return data

def sub_img(mrcData, x,y,boxsize):
    '''
    This part is to extract a box of pixels form origin mrc file
    '''
    box = np.empty((boxsize,boxsize))
    for row in range(boxsize):
        box[row,:] = mrcData[y][x:x+boxsize]
        y += 1
    return box

def read_particles(mic_path, dim_x, dim_y, boxsize, name_prefix, name_postfix, box_path, start_num, end_num,name_length,rotation_angel,rotation_n):
    particles = []
    for num in range(start_num, end_num + 1):
        boxX = []
        boxY = []
        mrc_name = mic_path + name_prefix + str(num).zfill(name_length) + name_postfix + ".mrc"
        box_name = box_path + name_prefix + str(num).zfill(name_length) + name_postfix + ".box"
        if not os.path.exists(mrc_name):
            print("%s is not exist!" % mrc_name)
            continue
        if not os.path.exists(box_name):
            print("%s is not exist!" % box_name)
            continue
        boxfile = open(box_name, 'r')
        for line in boxfile:
            col = line.split()
            x = int(col[0]) - 1
            y = int(col[1]) - 1
            if x < 0: x = 0
            if y < 0: y = 0
            if ( x  + boxsize > dim_x) or (y + boxsize ) > dim_y:
                continue
            boxX.append(x)
            boxY.append(y)
        boxfile.close()
        if len( boxX )  == 0:
            continue
        mrc = mrcfile.open(mrc_name)
        mrcstd = mapstd(mrc.data,dim_x,dim_y)
        particle = np.zeros((boxsize, boxsize))
        for ii in range(len(boxX)):
            index_x = boxX[ii]
            index_y = dim_y - boxY[ii] -boxsize
            particle = sub_img(mrcstd,index_x, index_y, boxsize)
            # rotate map
            image = Image.fromarray(particle)
            for i in range(rotation_n):
                image_rot = image.rotate(rotation_angel*i)
                # image_resize = image.resize(224,224)
                particles.append(np.array(image_rot))
        mrc.close()
    return particles

def load_train(args):
    '''
    This part include load training parameters and training data
    '''
    if not os.path.exists( args.mic_path) and os.path.exists(args.positive1_box_path ) and os.path.exists(args.negative1_box_path ) :
        print("Please make sure the mic path, positive1 path and negative1 path are exist!")
        exit -1

    positive1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, args.name_postfix, \
                                args.positive1_box_path, args.positive_mic_start_num, args.positive_mic_end_num, \
                                args.name_length,args.rotation_angel,args.rotation_n)
    negative1 = read_particles(args.mic_path, args.dim_x, args.dim_y, args.boxsize, args.name_prefix, args.name_postfix, \
                                args.negative1_box_path, args.negative_mic_start_num, args.negative_mic_end_num, \
                                args.name_length,args.rotation_angel,args.rotation_n)

    # shuffle positive1 and negative1
    random.shuffle(positive1)
    random.shuffle(negative1)
    positive1_particles_num = len(positive1)//args.rotation_n
    negative1_particles_num = len(negative1)//args.rotation_n

    print("positive1 size:" , positive1_particles_num)
    print("negative1 size:" , negative1_particles_num)
    if args.num_positive > positive1_particles_num:
        args.num_positive = positive1_particles_num
        print("positive1 only have %d particles, num_positive is set to %d" % (positive1_particles_num,positive1_particles_num))
    if args.num_negative > negative1_particles_num:
        args.num_negative = negative1_particles_num
        print("negative1 only have %d particles, num_negative2 is set to %d" % (negative1_particles_num, negative1_particles_num))

    train_size = int((args.num_positive + args.num_negative ) * args.rotation_n)
    test_size = int((args.num_p_test + args.num_n_test) * args.rotation_n)

    train_x = np.empty((train_size, args.boxsize, args.boxsize))
    train_y = np.zeros((train_size, 1))
    test_x = np.empty((test_size, args.boxsize, args.boxsize))
    test_y = np.zeros((test_size, 1))

    start = 0
    end = args.num_positive*args.rotation_n
    train_x[start:end] = positive1[0:args.num_positive*args.rotation_n]
    train_y[start:end] = 1

    start = end
    end += args.num_negative*args.rotation_n
    train_x[start:end] = negative1[0:args.num_negative*args.rotation_n]
    train_y[start:end] = 0

    if args.num_p_test > positive1_particles_num - args.num_positive:
        args.num_p_test = positive1_particles_num - args.num_positive 
        print("num_p_test is larger than the rest of positive particles, num_p_test is set to %d" % args.num_p_test)
    if args.num_n_test > negative1_particles_num - args.num_positive:
        args.num_n_test = negative1_particles_num - args.num_positive
        print ("num_n_test is larger than the rest of negative particles, num_n_test is set to %d" % args.num_n_test)

    start = 0
    end = args.num_p_test*args.rotation_n
    test_x[start:end] = positive1[args.num_positive*args.rotation_n:(args.num_positive + args.num_p_test)*args.rotation_n]
    test_y[start:end] = 1

    start = end
    end += args.num_n_test*args.rotation_n
    test_x[start:end] = negative1[args.num_negative*args.rotation_n:(args.num_negative + args.num_n_test)*args.rotation_n]
    test_y[start:end] = 0

    index = [i for i in range(len(train_x))]
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]


    # multi_gpu version
    train_len = len(train_y)
    test_len = len(test_y)

    train_num_batches = train_len // args.batch_size
    train_num_batch_gpus = train_num_batches//args.num_gpus

    test_num_batches = test_len // args.batch_size
    test_num_batch_gpus = test_num_batches // args.num_gpus

    train_x = train_x[0:args.num_gpus*args.batch_size*train_num_batch_gpus].reshape([-1,args.num_gpus, args.batch_size,args.boxsize, args.boxsize, 1])
    train_y = train_y[0:args.num_gpus*args.batch_size*train_num_batch_gpus].reshape([-1,args.num_gpus,args.batch_size,1])
    test_x = test_x[0:args.num_gpus*args.batch_size*test_num_batch_gpus].reshape([-1,args.num_gpus, args.batch_size,args.boxsize, args.boxsize, 1])
    test_y = test_y[0:args.num_gpus*args.batch_size*test_num_batch_gpus].reshape([-1,args.num_gpus,args.batch_size,1])

    return train_x, train_y, test_x, test_y

def non_max_suppression(particle,scores, boxsize, overlapThresh): 
    """Pure Python NMS baseline."""    
    x1 = particle[:,0]   
    y1 = particle[:,1]  
    x2 = x1 + boxsize
    y2 = y1 + boxsize
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)    
    pick = []    
    idxs = np.arange(len(scores))
   
    while scores.max() > 0:    
        index = np.argmax(scores)
        pick.append(index)  
        last = np.delete(idxs,np.concatenate(([index], np.where(scores < 0)[0])))
        xx1 = np.maximum(x1[index], x1[last])    
        yy1 = np.maximum(y1[index], y1[last])    
        xx2 = np.minimum(x2[index], x2[last])    
        yy2 = np.minimum(y2[index], y2[last])    
    
        w = np.maximum(0.0, xx2 - xx1 + 1)    
        h = np.maximum(0.0, yy2 - yy1 + 1)    

        # calculate the overlapping area 
        overlap = (w * h) / areas[last] 
        scores[np.concatenate(([index],last[np.where(overlap > overlapThresh)[0]]))] = -1
    return particle[pick].astype("int")

def load_predict(args, mrc_name):
    '''
    This part include load predict parameters and predict data
    '''
    test_x = []
    test_index = []
    mrc = mrcfile.open(mrc_name)
    mrc_std = mapstd(mrc.data,args.dim_x,args.dim_y)
    
    x_step_num = (args.dim_x - args.boxsize) // args.scan_step
    y_step_num = (args.dim_y - args.boxsize) // args.scan_step
    for i in range(x_step_num):
        for j in range(y_step_num):
            x = i*args.scan_step
            y = j*args.scan_step       
            img = sub_img(mrc_std,x, y, args.boxsize)
            # stddev = np.std(img)
            test_x.append(img)
            test_index.append([x,args.dim_y - args.boxsize - y])
    mrc.close()
        
    return test_x, test_index

if __name__ == '__main__':
    load_train()



















