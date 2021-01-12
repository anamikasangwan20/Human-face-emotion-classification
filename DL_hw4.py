r#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:42:09 2019

@author: anamika
"""
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#import os
#dataset_path = os.path.dirname(__file__)+'/DL_hw4/train_image/'
#
##Checking value and size of images:
#img = mpimg.imread(dataset_path + '00002.jpg')
#print(type(img))
#plt.imshow(img, cmap = 'gray')
#print(img.shape)
#print(np.max(img))
#print(np.min(img))
#
##Using Pillow to standardize images:
#from PIL import Image
#path = dataset_path
#i=1
#for item in os.listdir(path):
#    if os.path.isfile(path+item):
#        print (i)
#        i+=1
#        try:
#            img = Image.open(path+item)
#            img.verify()
#            img = Image.open(path+item)
#        except (IOError, SyntaxError, OSError) as e:
#            print ('Bad file: '+item)
#            continue
#        if img.mode != 'L':
#            img = img.convert(mode = 'L')
#        img.resize((350,350))
#        f, e = os.path.splitext(item)
#        img.save(path+'aug/'+f+'_bw.jpg',format='JPEG')
##Data Augmentation:
#        img_fliplr = img.transpose(Image.FLIP_LEFT_RIGHT)
#        img_fliptb = img.transpose(Image.FLIP_TOP_BOTTOM)
#        img_trans = img.transpose(Image.TRANSPOSE)
#        img_rot90 = img.rotate(90)
#        img_rot180 = img.rotate(180)
#        img_rot270 = img.rotate(270)
#        img_crop = img.crop((20,20,330,330))
#        img_crop = img_crop.resize((350,350))
#        img_fliplr.save(path+'aug/'+f+'_lr.jpg',format='JPEG')
#        img_fliptb.save(path+'aug/'+f+'_tb.jpg',format='JPEG')
#        img_trans.save(path+'aug/'+f+'_trans.jpg',format='JPEG')
#        img_rot90.save(path+'aug/'+f+'_90.jpg',format='JPEG')
#        img_rot180.save(path+'aug/'+f+'_180.jpg',format='JPEG')
#        img_rot270.save(path+'aug/'+f+'_270.jpg',format='JPEG')
#        img_crop.save(path+'aug/'+f+'_crop.jpg',format='JPEG')
#
##Labels:
#import os
#import numpy as np
#import glob
#from keras.utils import Sequence
##from keras.applications.resnet50 import ResNet50
#from PIL import Image
#import matplotlib.image as mpimg
#
#def process_image(imagefile):
#    im = Image.open(imagefile)
#    im = im.convert(mode = 'L')
#    im = im.resize((350,350))
#    # Add augmentation if you want
#    return np.asarray(im).reshape(350,350,1)
#
#class BatchSampler(Sequence):
#    '''
#    See https://keras.io/utils/
#    Sequences are a safer way of doing generators
#    '''
#    
#    def __init__(self, data_dir, label_file, batch_size):
#        '''
#        data_dir: train_image directory
#        label_file: train.csv
#        '''
#        self.batch_size = batch_size
#        self.image_files = glob.glob(data_dir+'/*.jpg') #get all files in data directory
#        self.labels_dict = {}
#        self.label2id = {}
#        
#        # get all labels
#        lines = open(label_file).read().splitlines()
#        labels = []
#        for line in lines:
#            name,label = line.split(',')
#            self.labels_dict[name] = label
#            labels.append(label)
#        
#        # create integer labels for each emotion
##        count=0
##        for l in set(labels):
##            self.label2id[l] = count
##            count+=1
#        self.label2id['anger'] = 0
#        self.label2id['contempt'] = 1
#        self.label2id['disgust'] = 2
#        self.label2id['fear'] = 3
#        self.label2id['happiness'] = 4
#        self.label2id['neutral'] = 5
#        self.label2id['sadness'] = 6
#        self.label2id['surprise'] = 7
#        
#    def __len__(self):
#        '''
#        number of batches in 1 epoch
#        '''
#        return int(np.ceil(len(self.image_files) / float(self.batch_size)))
#
#    def __getitem__(self, idx):
#        '''
#        This is like the generator
#        Outputs a batch of samples
#        '''
#        batch_x = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
#        batch_y = []
#        for file in batch_x:
#            file_name = file.split('/')[-1] #this isolates only the filename
#            label = self.labels_dict[file_name] 
#            batch_y.append(self.label2id[label])
#        return np.array([process_image(file) for file in batch_x]), np.array(batch_y)
#
##datapath = os.path.dirname(__file__)+'/DL_hw4/train_image/'
#bs = BatchSampler(os.path.dirname(__file__)+'/DL_hw4/train_image', os.path.dirname(__file__)+'/DL_hw4/train.csv', 32)
#print(len(bs.image_files))
#print(bs.label2id)
#print(len(bs))    
#
#Data Save:
#class BatchSampler2(Sequence):
#
#    def __init__(self, data_dir, batch_size):
#
#        self.batch_size = batch_size
#        #self.image_files = glob.glob(data_dir+'/*.jpg') #get all files in data directory
#        self.all_images = []
#        self.labels = []
#        #i=1
#        j=0
#        for item in os.listdir(data_dir):
#            try: 
#                img = mpimg.imread(data_dir+item)
#                self.all_images = np.append(self.all_images,img)
#                self.labels = np.append(self.labels,bs.labels_dict[item[:5]+'.jpg'])
#                
#                print (len(np.array(self.labels)))
#            except (ValueError):
#                j+=1
#                continue
#        print ('Number of discarded samples due to shape: '+str(j))
#            
#        
#
#
#
#bs2 = BatchSampler2(os.path.dirname(__file__)+'/DL_hw4/train_image/aug/', 300)



#___________________________________________________________________

#import os
##from skimage import io
#import numpy as np
#import matplotlib.image as mpimg
##from PIL import Image 
#
#datacnn_path = os.path.dirname(__file__)+'/DL_hw4/train_image/aug/'
#
#datapath = os.path.dirname(__file__)+'/DL_hw4/train_image/'
##datacnn_path = 'https://s3.amazonaws.com/ee599hw4weights/images/'
##loaded = np.load(datacnn_path)
##all_images = np.zeros((4,350,350,1))
#all_images = []
#labels = []
#batch1_images = []
#batch2_images = []
#batch3_images = []
#batch1_size = 300
#batch2_size = 20
#batch1=1
#batch2=1
#i=1
#j=0
#for item in os.listdir(datacnn_path):
#      #img = Image.open(datacnn_path+item)
#      #img = io.imread(datacnn_path+item, as_grey=True)
#      #img = io.imread(item, as_grey=True)
#      print(batch1_size*batch2_size*(batch2-1) + batch1_size*(batch1-1) + i)  #-------------??????????    
#      img = mpimg.imread(datacnn_path+item)
#      try:
#          img = np.reshape(img,(350, 350, 1))
#          batch1_images = np.append(batch1_images,img)
#          emotion = bs.labels_dict[item[:5]+'.jpg']
#          labels = np.append(labels,bs.label2id[emotion])
#          #labels = np.dtype(labels, np.int16)
#          i+=1
#          #print (item)
#          if i > batch1_size:
#              batch2_images = np.append(batch2_images,batch1_images)
#              batch1_images = []
#              i=1
#              batch1+=1
#          if batch1 > batch2_size:
#             #print ('Check size: batch2_images has: '+str(len(batch2_images))+' = '+str(batch1_size*batch2_size*350*350))
#              batch2_images = np.reshape (batch2_images,(batch1_size*batch2_size,350,350,1))
#              np.savez_compressed (datapath+'cnn_batchtry'+str(batch2),batch2_images)
#              print ('batch2 is: '+str(batch2))
#              #batch3_images = np.append(batch3_images,batch2_images)
#              batch1_images = []
#              batch2_images = []
#              #i=1
#              batch1=1
#              batch2+=1
#              
#              
#      except (ValueError):
#          #print ('Value Error: Shape of '+item+' is '+str(img.shape))
#          j+=1
#          continue
##x_train = np.array(all_images)
##if len(batch2_images)>0:
##    batch2_images = np.append(batch2_images,batch2_images)
#if len(batch1_images)>0:
#    batch2_images = np.append(batch2_images,batch1_images)
#    print ('Leftover data values are: '+str(batch1_size*(batch1-1) + (i-1))+' in number stored in cnn_batch'+str(batch2))
#    batch2_images = np.reshape (batch2_images,((batch1_size*(batch1-1) + (i-1)),350,350,1))
#    np.savez_compressed (datapath+'cnn_batch'+str(batch2),batch2_images)
#
#print ('Number of discarded samples for shape: '+str(j))
#_______________________________________________________________________
##For labels:
#import os
#import numpy as np
#import matplotlib.image as mpimg
#
#
#datacnn_path = os.path.dirname(__file__)+'/DL_hw4/train_image/aug/'
#
#datapath = os.path.dirname(__file__)+'/DL_hw4/train_image/'
#labels = []
#i=1
#j=0
#for item in os.listdir(datacnn_path):
#      print(i)
#      img = mpimg.imread(datacnn_path+item)
#      try:
#          img = np.reshape(img,(350, 350, 1))
#          emotion = bs.labels_dict[item[:5]+'.jpg']
#          labels = np.append(labels,bs.label2id[emotion])
#          i+=1
#      except (ValueError):
#          #print ('Value Error: Shape of '+item+' is '+str(img.shape))
#          j+=1
#          continue
#
#print('Length of labels: '+str(len(labels)))
#print ('Number of discarded samples for shape: '+str(j))
#labels = np.reshape (labels, (len(labels),1))
#np.savez_compressed (datapath+'labels',labels)
##_______________________________________________________________________    

#Loading npz:
import os
import numpy as np
datapath = os.path.dirname(__file__)+'/DL_hw4/train_image/'
all_images1 = []
all_images2 = []
all_images3 = []
all_images = []

#for i in range(5):
#    loaded = np.load(datapath+'cnn_batch'+str(i+1)+'.npz')
#    all_images1 = np.append(all_images1,loaded['arr_0'])
#    print (i)
#for i in range(5,10):
#    loaded = np.load(datapath+'cnn_batch'+str(i+1)+'.npz')
#    all_images2 = np.append(all_images2,loaded['arr_0'])
#    print (i)
#for i in range(10,17):
#    loaded = np.load(datapath+'cnn_batch'+str(i+1)+'.npz')
#    all_images3 = np.append(all_images3,loaded['arr_0'])
#    print (i)

loaded1 = np.load(datapath+'cnn_batch1.npz')
print (1)
loaded2 = np.load(datapath+'cnn_batch2.npz')
print (2)
loaded3 = np.load(datapath+'cnn_batch3.npz')
print (3)
loaded4 = np.load(datapath+'cnn_batch4.npz')
print (4)
loaded5 = np.load(datapath+'cnn_batch5.npz')
print (5)
loaded6 = np.load(datapath+'cnn_batch6.npz')
print (6)
loaded7 = np.load(datapath+'cnn_batch7.npz')
print (7)
loaded8 = np.load(datapath+'cnn_batch8.npz')
print (8)
loaded9 = np.load(datapath+'cnn_batch9.npz')
print (9)
loaded10 = np.load(datapath+'cnn_batch10.npz')
print (10)
loaded11 = np.load(datapath+'cnn_batch11.npz')
print (11)
loaded12 = np.load(datapath+'cnn_batch12.npz')
print (12)
loaded13 = np.load(datapath+'cnn_batch13.npz')
print (13)
loaded14 = np.load(datapath+'cnn_batch14.npz')
print (14)
loaded15 = np.load(datapath+'cnn_batch15.npz')
print (15)
loaded16 = np.load(datapath+'cnn_batch16.npz')
print (16)
loaded17 = np.load(datapath+'cnn_batch17.npz')
print (17)

#all = []
#
#all = np.concatenate((loaded1['arr_0'],loaded2['arr_0'],loaded3['arr_0']))
#print('all concatenated')
#n=6000*3
#all = np.reshape(all,(n,350,350,1))
#print('all reshaped')
#np.savez_compressed (datapath+'cnn_18000',all)
#print('all saved')
#
#all = []
#
#all = np.concatenate((loaded4['arr_0'],loaded5['arr_0'],loaded6['arr_0']))
#print('all concatenated')
#n=6000*3
#all = np.reshape(all,(n,350,350,1))
#print('all reshaped')
#np.savez_compressed (datapath+'cnn_18000_2',all)
#print('all saved')

#all = []
#
#all = np.concatenate((loaded7['arr_0'],loaded8['arr_0'],loaded9['arr_0']))
#print('all concatenated')
#n=6000*3
#all = np.reshape(all,(n,350,350,1))
#print('all reshaped')
#np.savez_compressed (datapath+'cnn_18000_3',all)
#print('all saved')
#
all = []

all = np.concatenate((loaded10['arr_0'],loaded11['arr_0'],loaded12['arr_0'],loaded13['arr_0']))
print('all concatenated')
n=6000*4
all = np.reshape(all,(n,350,350,1))
print('all reshaped')
np.savez_compressed (datapath+'cnn_24000_1013',all)
print('all saved')

#loaded18 = np.load(datapath+'cnn_18000.npz')
#print (18)
#loaded19 = np.load(datapath+'cnn_18000_2.npz')
#print (19)
#
#all = []
#all = np.concatenate((loaded18['arr_0'],loaded19['arr_0']))
#print('all concatenated')
#n=18000*2
#all = np.reshape(all,(n,350,350,1))
#print('all reshaped')
#np.savez_compressed (datapath+'cnn_36000_1',all)
#print('all saved')

#nn=97797
#all_images = np.append(all_images,all_images1)
#print('all_images1 appended')
#all_images = np.append(all_images,all_images2)
#print('all_images2 appended')
#all_images = np.append(all_images,all_images3)
#print('all_images3 appended')
#
#all_images = np.reshape(all_images,(nn,350,350,1))
#print('all_images reshaped')
#np.savez_compressed (datapath+'cnn',all_images)
#print('all_images saved')


#len(loaded['arr_0'])
#Out[35]: 500
#
#len(loaded)
#Out[36]: 1
#
#len(loaded['arr_0'][0])
#Out[37]: 350
#
#len(loaded['arr_0'][0][0])
#Out[38]: 350


#CNN









