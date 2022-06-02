# -*- coding: utf-8 -*-
"""
Created on sat April 18 10:09:27 2020
@author: Dreamen
"""
#%%
import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
from tensorflow import keras
import keras.backend.tensorflow_backend as KTF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import Sequence
from skimage.transform import resize
import cv2
import math
import SimpleITK
from imgaug import augmenters as iaa
#import keras
from keras.applications.vgg16 import vgg16,preprocess_input
from keras import layers, metrics
from keras.layers import Input, Dropout,Flatten,Dense
#from keras.layers.core import Dense
from keras.models import Model,Sequential
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping,CSVLogger
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from lifelines.utils import concordance_index
import sys
import copy
from keras.utils import multi_gpu_model
# sys.path.append("/data/zlw/survival1/code/")
# import resnet
sys.path.append("/data/zlw/survival1/TMImultitask/code/")
# from SurvmodelTMI2 import resnet_self
from attentionNet1117debug import resnet_self
#from selfResNeXt import ResNeXt
from keras.metrics import binary_accuracy
from keras.utils import plot_model
import time
date1 = time.strftime('%m%d%H%M',time.localtime(time.time()))
import scipy.misc
clinical_features = ['age','gender','cTstage','cNstage','cTNMstage']
clinical_features = []

#%%
def calc_at_risk(X, T, O):
    '''
    # function description in survivalnet risklayer
#    tmp = list(T)
#    T = np.asarray(tmp).astype('float64')
    '''
    order = np.argsort(T.astype('float64'))
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')
#    T = np.asarray(sorted_T)
    O = O[order]
    X = X[order]
    return X, O, sorted_T, at_risk
class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support
    For more information of imgaug see: https://github.com/aleju/imgaug
    """
    def __init__(self, dataset_csv_file, source_image_dir, batch_size=16,
                 target_size=(224, 224), verbose=0, steps=None,label_cTNM=3,labelTNM=3,
                 mode='train', random_state=1, flag =0):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        dataset_df = pd.read_csv(dataset_csv_file)
        #remove guizhou data
        #dataset_df1 = dataset_df[dataset_df.flag!=1]
        #dataset_df1.to_csv('/data/zlw/survival1/TMImultitask/csv/removeliwuchao.csv',header = True)
        self.dataset_df = dataset_df[dataset_df['flag'] ==flag].copy()
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.mode = mode
        self.shuffle = True if self.mode=='train' else False
        self.random_state = random_state
        self.prepare_dataset()
        #add multi
        self.TNM2id={'I':0,'II':1,'III':2}
        self.clabelTNM = labelTNM
        # self.indices = np.arange(len(dataset_df['hospital']))
        # for k in clinical_features:
        #     if dataset_df[k].ndim==1:
        #             dataset_df1[f'{k}'] = np.expand_dims(dataset_df[k],axis=1)

        # self.clinical_feats = np.concatenate([dataset_df1[k] for k in clinical_features],axis=1)#[self.indices]
        # if len(np.where(np.isnan(self.clinical_feats)))>0:
        #     print(np.where(np.isnan(self.clinical_feats)))

        if steps is None:
            self.steps = math.ceil(self.x_path.shape[0] / float(self.batch_size))
        else:
            self.steps = int(steps)
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(dcm_file, seg_file, ind)[0] for dcm_file, seg_file, ind in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_x2 = np.asarray([self.load_image(dcm_file, seg_file, ind)[1] for dcm_file, seg_file, ind in batch_x_path])
        batch_x2= self.transform_batch_images(batch_x2)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_classTNM = self.classTNM[idx * self.batch_size:(idx + 1) * self.batch_size]#multi
        batch_classT =   self.classT[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_classN =   self.classN[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(clinical_features)>0:
            batch_clinical_feats = self.clinical_feats[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.mode=='train':
            batch_x, *batch_y_ = calc_at_risk(batch_x, batch_y[:, 1], batch_y[:, 0])#* denote to separate last two variables
            batch_y = np.vstack(batch_y_).T
        else:
            at_risk = np.asarray([list(np.sort(batch_y[:,1])).index(x) for x in batch_y[:,1]], dtype='int32')
            batch_y = np.hstack((batch_y, at_risk.reshape(at_risk.shape+(1,))))
        if len(clinical_features)>0:
            return [batch_x,batch_x2,batch_clinical_feats], [batch_y,batch_classT,batch_classN]
        else:
            return [batch_x,batch_x2], [batch_y,batch_classT,batch_classN]

    def load_image(self, dcm_file, seg_file, ind):
        dcm_path = os.path.join(self.source_image_dir, dcm_file)
        seg_path = os.path.join(self.source_image_dir, seg_file)
        dcm = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(dcm_path))[ind, :, :]
        seg = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(seg_path))[ind, :, :]
        #plt.imshow(seg)#
#        (ystart, xstart), (ystop, xstop)= boundingBox( seg, use2D=True)
        # dcmcopy = copy.deepcopy(dcm)
        # seg = np.uint8(seg)
        # contours1, hierarchy1 = cv2.findContours(seg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # # 如果最后一个参数是-1，则填充勾画区域，如果是数字，则控制厚度
        # img1 = cv2.drawContours(dcmcopy,contours1,-1,(22,22,22),-1)
        # diff = dcm - img1 # set backgroud to zero
        pos_x, pos_y = np.where(seg==1)
#        image = dcm[xstart: xstop, ystart:ystop].astype(float)
        # image = diff[pos_x.min():pos_x.max()+1, pos_y.min():pos_y.max()+1].astype(float)
        image = dcm[pos_x.min():pos_x.max()+1, pos_y.min():pos_y.max()+1].astype(float)
        #path = '../da
        #print('dcm_file:',dcm_file)
#        [dirname,tempfilename]= os.path.split(dcm_file)
#        (filename1, extension) = os.path.splitext(tempfilename)
#        scipy.misc.imsave(f'../data/imageROI/{filename1}.jpg', image)
        # plt.imshow(image)
        image_array = (image-image.min()) / (image.max()-image.min())
        image_array = np.stack((image_array, image_array, image_array), -1)
        image_array = cv2.resize(image_array, self.target_size,interpolation=cv2.INTER_CUBIC)
        # image_array = resize(image_array, self.target_size,mode ='constant')
        #for input 2 
        kk = 0.1
        KKx = int((np.abs(pos_x.max()-pos_x.min()))*kk)
        KKy = int((np.abs(pos_y.max()-pos_y.min()))*kk)
        image2 = dcm[pos_x.min()-KKx:pos_x.max()+1+KKx, pos_y.min()-KKy:pos_y.max()+1+KKy].astype(float)

        #path = '../da
        #print('dcm_file:',dcm_file)
#        [dirname,tempfilename]= os.path.split(dcm_file)
#        (filename1, extension) = os.path.splitext(tempfilename)
#        scipy.misc.imsave(f'../data/imageROI/{filename1}.jpg', image)
        # plt.imshow(image)
        image_array2 = (image2-image2.min()) / (image2.max()-image2.min())
        image_array2 = np.stack((image_array2, image_array2, image_array2), -1)
        image_array2 = cv2.resize(image_array2, self.target_size,interpolation=cv2.INTER_CUBIC)       
        return image_array,image_array2
    def transform_batch_images(self, batch_x):
        if self.mode=='train':
            augmenter = iaa.Sequential(
                [
                    iaa.Fliplr(0.2),
                    iaa.Flipud(0.3),
                    iaa.Sometimes(0.3,
                        iaa.SomeOf(1,[
                            iaa.Affine(rotate=(-10,10),cval=0,mode='constant'),
                            iaa.Affine(translate_px=(-10,10),cval=0,mode='constant'),
                            iaa.Crop(percent=(0, 0.1)),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            #                iaa.Crop(percent=(0, 0.1)),
            #                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                        ])),
                ],
                random_order=True,
            )
            batch_x = augmenter.augment_images(batch_x)
#        imagenet_mean = np.array([0.485, 0.456, 0.406])
#        imagenet_std = np.array([0.229, 0.224, 0.225])
#        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.
        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]# for c-index
    def get_image_index(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.x_path[:self.steps*self.batch_size]
    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y = df[['dcm_path', 'seg_path', 'slice']].values, df[['event', 'time']].values
        self.classTNM = df[['cTNMstage01']].values
        self.classT = df[['cTstage01']].values
        self.classN = df[['cNstage02']].values
        dataset_df1 = {}
        # for k in clinical_features:
        #     if df[k].ndim==1:
        #             dataset_df1[f'{k}'] = np.expand_dims(df[k],axis=1)
        # self.clinical_feats = np.concatenate([dataset_df1[k] for k in clinical_features],axis=1)#[self.indices]
        df['age'].fillna(value=62,inplace=True)
        self.clinical_feats = df[['age', 'gender', 'cTNMstage01','cTstage01','cNstage01']].values
        # self.classTNM =  keras.utils.to_categorical(df[['cTstage01']].values)
    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

class CalCindex(Callback):
   """
   Monitor mean AUROC and update model
   """
   def __init__(self, dataset_csv_file, source_image_dir, batch_size=16,
                 target_size=(224, 224), verbose=0,random_state=1, workers=1,num2 = 0, log_path='.\\'):
      super(Callback, self).__init__()
      self.train_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
               target_size=target_size, verbose=verbose, steps=None,
               mode='test', random_state=random_state, flag = 0)
      self.val_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
               target_size=target_size, verbose=verbose, steps=None,
               mode='test', random_state=random_state, flag = 2)
      self.test_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
               target_size=target_size, verbose=verbose, steps=None,
               mode='test', random_state=random_state, flag = 2)
      self.workers = workers
      self.log_path = os.path.join(log_path, 'cindex.csv')
      self.num2 = num2
   def on_epoch_end(self, epoch, logs={}):
      """
      Calculate the average AUROC and save the best model weights according
      to this metric.

      """
    #   print("\n*********************************")
        #show learning rate
      self.lr = float(K.eval(self.model.optimizer.lr))
      self.val_loss = logs.get('val_loss')
      print(f"current learning rate: {self.lr:.5f}")
#        print("current learning rate: %d",% self.lr) for python 3.5
      """
      y_hat shape: (#samples, len(class_names))
      y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
      """
      cindex = []
      for i,se in enumerate([self.train_sequence, self.val_sequence,self.test_sequence]):#,self.test_sequence
          y_hat = self.model.predict_generator(se, workers=self.workers)
          y = se.get_y_true()
          path = se.get_image_index()
          pid = np.array([p.split('/')[-2] for p in path[:, 0]])
          res_all = pd.DataFrame(dict(PatientID=pid, Event=y[:, 0], Time=y[:, 1], Risk=y_hat[0].squeeze(),RiskTstage=y_hat[1].squeeze(),RiskNstage=y_hat[2].squeeze()))
          res = pd.DataFrame([[pid_, group.Event.mean(), group.Time.mean(), group.Risk.mean(), group.RiskTstage.mean(), group.RiskNstage.mean()] for pid_, group in res_all.groupby(by='PatientID')], columns=list(res_all))
        #   res_all.to_csv(os.path.join(saveriskpath,f'all_{epoch}_{i}.csv'), index=False)
          if model_weights_file:
                res.to_csv(os.path.join(saveriskpath,f'{savename}_{epoch}_{i}.csv'), index=False)
          cindex.append(round(1-concordance_index(res.Time, res.Risk, res.Event),4))
      print(f"*** epoch#{epoch + 1} dev ***")
      print(f"cindex: {cindex}")
      print(f'val_loss:{self.val_loss:.4f}')
      print("******************************************************************************")
      with open(self.log_path, "a") as f:
          f.write(f"(epoch#{epoch + 1}) cindex: {cindex}, lr: {self.lr:.5f}, val_loss:{self.val_loss:.4f}\n")
      return
#%%model 2 resnet 18
def model_icc(input_shape):
#    img_input = Input(shape=input_shape)
    base_model = resnet.ResnetBuilder.build_resnet_18(input_shape, num_outputs=1)
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    base_model.output_layers = [base_model.layers[-1]]
    x = base_model.layers[-1].output
    risk_pred = Dense(1, activation="sigmoid", name="risk_pred", use_bias=False)(x)
    model = Model(inputs=base_model.input, outputs=risk_pred)
    # model.summary()
    return model
    # print(model.summary())
#    plot_model(model, to_file='./model20-1.pdf',show_shapes=True)
def loss_icc(y_true, y_pred):#y_truebservation
    # K.print_tensor('y_pred is :',y_pred)
    exp = K.exp(y_pred)[::-1] ##xiaohan origin
    partial_sum = K.cumsum(exp)[::-1]
#    log_at_risk = K.log(K.gather(partial_sum, K.cast(y_true[:, 2], dtype='int32')))
    log_at_risk = K.log(partial_sum)#+ y_pred.max()
    diff = y_pred - log_at_risk
    cost = -K.sum(y_true[:,0]*diff)/(K.sum(y_true[:,0])+1e-6)
    return cost
#%%
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    os.chdir("/data/zlw/survival1/code")
    clinical_features = ['age','gender','cTstage','cNstage','cTNMstage']
    clinical_features =[]
    savename = 'test3'
    output_dir = '../TMImultitask/h5TMI/'+ f'{date1}{savename}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    saveriskpath = '../TMImultitask/csv/riskattention'
    if not os.path.exists(saveriskpath):
        os.mkdir(saveriskpath)
    source_image_dir = '/data/zlw/survival1/data/data3sliceall3'
    dataset_csv_file = '../TMImultitask/csv/onlygastrictrainandval823.csv'

    dataset_csv_file = '/data/zlw/survival1/TMImultitask/csv/onlygastrictrainandval823addwuchaodata329.csv'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    dataset = pd.read_csv(dataset_csv_file)
    print("** create image generators **")
    batch_size =8# 8
    lr = 0.0001
    epochnum = 300
    train_steps = math.ceil(dataset[dataset.flag==0].shape[0] / batch_size)
    input_shape = 224
    train_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
              target_size=(input_shape, input_shape), verbose=1, steps=train_steps, mode='train', flag = 0)
    val_sequence = AugmentedImageSequence(dataset_csv_file, source_image_dir, batch_size=batch_size,
                     target_size=(input_shape, input_shape), verbose=1, steps=None,
                     mode='test', flag = 2)
    
    input_shape3 = (input_shape, input_shape, 3)
    model =  resnet_self(input_shape = input_shape3)   ###############
    # model = model_icc(input_shape3)
    # para_model = multi_gpu_model(model,gpus=2)
    # print(model.summary())
#        model = model_icc1((input_shape, input_shape, 3))
    model_weights_file =None#'/data/zlw/survival1/TMImultitask/h5TMI/03121942self/res.205-9.2863.h5'# '/data/zlw/survival/modelweight811/676575.h5'
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5load/2080T1AttCBAM6568.h5'#foe CBAM attention block 1222
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/11242301/res.139-8.3402.h5'#666460
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5load/attention705705best.h5'
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/03160835selfk0.1/res.162-8.2371.h5'#7574
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/03160835selfk0.1/res.151-8.1859.h5'
    model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/03231124selfk0.3/res.144-8.6104.h5'
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/03281644selfk0.4/res.217-8.4426.h5'
    # model_weights_file = '/data/zlw/survival1/TMImultitask/h5TMI/03290931selfk0.2/res.116-7.9512.h5' #7174 final
    


    if model_weights_file:
        model.load_weights(model_weights_file)
        epochnum = 1
        for layer in model.layers: #total 0-66[:45]
             layer.trainable = False
    # for layer in model.layers:
    #     layer.trainable = False
    # print(model.summary())                     
    Adamop = Adam(lr=lr,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    SGD = SGD(lr=lr, decay=1e-5, momentum=0.19, nesterov=True)
    print("** compile model with class weights **")
    '''
    model.compile(optimizer=SGD, loss={'risk_pred':loss_icc,'TNM_stage':'mean_squared_error'},#,'TNM_stage':'categorical_crossentropy'
            loss_weights={'risk_pred': 0.2, 'TNM_stage': 0.8},metrics=['accuracy'])#{'TNM_stage':keras.metrics.categorical_accuracy}
    '''
    model.compile(optimizer=SGD, loss={'risk_pred':loss_icc,'T_stageOutput':'binary_crossentropy','N_stageOutput':'binary_crossentropy'},#,'TNM_stage':'categorical_crossentropy'
            loss_weights={'risk_pred': 1, 'T_stageOutput': 1,'N_stageOutput':1},metrics=['accuracy'])
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'res.{epoch:02d}-{val_loss:.4f}.h5'),
        save_weights_only=True,save_best_only=False,verbose=1)
    cal_cindex = CalCindex(dataset_csv_file, source_image_dir, batch_size=batch_size,
                    target_size=(input_shape, input_shape), log_path=output_dir)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger(os.path.join(output_dir, 'training.csv'))
    callbacks = [
            checkpoint,
            # TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
            #                   verbose=1, mode="min", min_lr=1e-7),
            cal_cindex,
            csv_logger
            # early_stopping,
                ]
    print("** start training **")
    model.fit_generator(
            generator=train_sequence,
            steps_per_epoch=train_steps,
            epochs=epochnum,verbose=2,
            validation_data=val_sequence,
            validation_steps=math.ceil(dataset[dataset.flag==1].shape[0] / batch_size),
            callbacks=callbacks, 
            workers=16,
            shuffle=False,
        )