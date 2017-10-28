#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:36:45 2017

@author: no1
"""

import tensorflow as tf  
import numpy  
import scipy.misc as misc
import os
import cv2
def write_binary(filename):  
    cwd = os.getcwd()
    output_path=os.path.join(cwd,'datasets','man2woman',filename)
    dirname=os.path.dirname(output_path)
    dirname=os.path.join(dirname,'a_resized')
    writer = tf.python_io.TFRecordWriter(output_path)  

    for img_name in os.listdir(dirname):
        img_path = os.path.join(dirname , img_name)
              
        with tf.gfile.FastGFile(img_path, 'rb') as f:
            img_raw = f.read()         
 
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/file_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_name)])),
                'image/encoded_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
            ))
           
    #序列化  
        serialized = example.SerializeToString()  
    #写入文件  
        writer.write(serialized)  
        
    writer.close()  


    
def read_and_decode(filename):  
    #创建文件队列,不限读取的数量  
    filename_queue = tf.train.string_input_producer([filename],shuffle=False)  
    # create a reader from file queue  
    reader = tf.TFRecordReader()  
    #reader从文件队列中读入一个序列化的样本  
    _, serialized_example = reader.read(filename_queue)  
  
    features = tf.parse_single_example(  
        serialized_example,  
        features={  
            'image/file_name': tf.FixedLenFeature([], tf.string),  
            'image/encoded_image': tf.FixedLenFeature([], tf.string) 
            
        }  
    )  
    img=tf.image.decode_jpeg(features['image/encoded_image'],channels=3)
    img = tf.reshape(img, [256, 256, 3])
  
    return img  


#write_binary('x.tfrecords')  
#%%
tfrecord='datasets/man2woman/man.tfrecords'
img = read_and_decode(tfrecord) 
img_batch = tf.train.shuffle_batch([img], batch_size=28, capacity=1003, min_after_dequeue=1000, num_threads=8)  
  

init = tf.global_variables_initializer() 
sess = tf.Session()  
 
sess.run(init)  
coord = tf.train.Coordinator()  
threads=tf.train.start_queue_runners(sess=sess,coord=coord)  

img = sess.run(img_batch)
for i in range(18):   
    [b,g,r]=[cv2.split(img[i])[0],cv2.split(img[i])[1],cv2.split(img[i])[2]]
    cv2.imwrite('{}.jpg'.format(i),cv2.merge([r,g,b]))
coord.request_stop()
coord.join(threads)
sess.close()
#%%
