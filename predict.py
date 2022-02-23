import pickle
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import os
from tensorflow.keras import regularizers
import datetime
import sys

if  len(sys.argv)!=2:
    print("Usage: python predict.py [dataset]")
    exit(1)



data_dir="PPIdata/"+str(sys.argv[1])+"/"

#Read the protein sequences
seqFile=open(data_dir+"sequenceList.txt")
seq=seqFile.readlines()
seqFile.close()

maxlen=5000
enseq=[]
for s in seq:
    x=[ord(i)-ord('A')+1 for i in s.rstrip('\n')]
    x_len=len(x)
    x=x+[0]*(maxlen-x_len)
    x=x[0:maxlen]
    enseq.append(x)

# Make test data set

ppiset=pd.read_table(data_dir+'test.txt',sep='\t',header=None)


TestSeqA=[]
TestSeqB=[]
for i in range(len(ppiset)):
    TestSeqA.append(enseq[ppiset.loc[i,0]])
    TestSeqB.append(enseq[ppiset.loc[i,1]])


class ConvPool(tf.keras.layers.Layer):
    def __init__(self, numFilters, kernelSize, GlobalPool=False, **kwargs):
        super(ConvPool,self).__init__(**kwargs)
        self.convLayer=tf.keras.layers.Conv1D(filters=numFilters,
                                              kernel_size=kernelSize,
                                              padding='same',
                                              kernel_regularizer=regularizers.l2(0.1),
                                              bias_regularizer=regularizers.l2(0.1))
        self.relu=tf.keras.layers.ReLU()
        self.batchNorm=tf.keras.layers.BatchNormalization()
        if GlobalPool:
            self.pooling=tf.keras.layers.GlobalAveragePooling1D()
        else:
            self.pooling=tf.keras.layers.AveragePooling1D()

    def call(self, inputSeq,training):
        inputSeq=self.convLayer(inputSeq)
        inputSeq=self.relu(inputSeq)
        inputSeq=self.batchNorm(inputSeq,training=training)
        inputSeq=self.pooling(inputSeq)
        return inputSeq


class PPIhashModel(tf.keras.models.Model):
    def __init__(self, numOfToken, embDim, trainEmbLayer, embWeights=None, **kwargs):
        super(PPIhashModel,self).__init__()
        if embWeights is None:
            self.embedding = tf.keras.layers.Embedding(input_dim=numOfToken,
                                                       output_dim=embDim,
                                                       trainable=trainEmbLayer)
        else:
            self.embedding=tf.keras.layers.Embedding(input_dim=numOfToken,
                                     output_dim=embDim,
                                     trainable=trainEmbLayer,
                                     weights=[embWeights])
        self.randomProjectOne=tf.keras.layers.Dense(
                              units=64,activation=None,use_bias=False,
                              trainable=False,kernel_regularizer=regularizers.l2(0.1),name="RandomProjectionOne")

        self.randomProjectTwo=tf.keras.layers.Dense(
                              units=64,activation=None,use_bias=False,
                              trainable=False,kernel_regularizer=regularizers.l2(0.1),name="RandomProjectionTwo")
        self.ConvBlock1=ConvPool(numFilters=64,kernelSize=5,
                                GlobalPool=False,name="ConvBlock1")
        self.ConvBlock2=ConvPool(numFilters=128,kernelSize=7,
                                GlobalPool=False,name="ConvBlock2")
        self.ConvBlock3=ConvPool(numFilters=256,kernelSize=9,
                                GlobalPool=False,name="ConvBlock3")
        self.ConvBlock4=ConvPool(numFilters=512,kernelSize=15,
                                GlobalPool=True,name="ConvBlock4")

    @tf.function # mark the function for compilation
    def call(self,inputSeqA,inputSeqB,training):
        """
        @brief The feed-forward logic of the network
        @param inputTensorA: tensor, the numerically encoded tensor for the first protein.整数序列，补零至1000
        @param inputTensorB: tensor, the numerically encoded tensor for the second protein.
        """
        proteinA=self.embedding(inputSeqA)
        proteinB=self.embedding(inputSeqB)

        proteinA=self.ConvBlock1(proteinA,training=training)
        proteinB=self.ConvBlock1(proteinB,training=training)
        proteinA=self.ConvBlock2(proteinA,training=training)
        proteinB=self.ConvBlock2(proteinB,training=training)
        proteinA=self.ConvBlock3(proteinA,training=training)
        proteinB=self.ConvBlock3(proteinB,training=training)
        proteinA=self.ConvBlock4(proteinA,training=training)
        proteinB=self.ConvBlock4(proteinB,training=training)

        proteinA=self.randomProjectOne(proteinA)
        proteinA=tf.keras.activations.sigmoid(proteinA)
        proteinB=self.randomProjectTwo(proteinB)
        proteinB=tf.keras.activations.sigmoid(proteinB)
        return proteinA,proteinB


numTokens=27
embDim=8
trainEmbedding=1
gpuIndex=0

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuIndex)
for gpu in tf.config.experimental.get_visible_devices()[1:]:
    tf.config.experimental.set_memory_growth(gpu,True)

model=PPIhashModel(numTokens,embDim,trainEmbedding)
optimizer=tf.keras.optimizers.Adam()
lossFunction=tf.keras.losses.BinaryCrossentropy()

## construct evaluation metrics for training and Test datasets:
# Training metrics:
trainLoss=tf.keras.metrics.Mean(name="trainLoss") 
trainBasicloss=tf.keras.metrics.Mean(name="trainBasicloss")
trainHashloss=tf.keras.metrics.Mean(name="trainHashloss")
trainBitloss=tf.keras.metrics.Mean(name="trainBitloss")

# define the input signature:
InputSignature=[
         tf.TensorSpec(shape=(None,len(TestSeqA[0])),
                       dtype=tf.int32),
         tf.TensorSpec(shape=(None,len(TestSeqA[0])),
                       dtype=tf.int32),
         tf.TensorSpec(shape=(None,), dtype=tf.int32)
        ]

#The hash loss
@tf.function
def basicLoss(label, dist,BATCH_SIZE):
    basicloss = 0.0
    for i in range(BATCH_SIZE):
        if label[i]==1:
            basic_loss= tf.maximum(dist[i] - 2, 0.0)
        else:
            basic_loss= tf.maximum(12 - dist[i], 0.0)
        basicloss = basicloss + basic_loss
    mean_basicloss = basicloss/BATCH_SIZE
    return mean_basicloss


def hashLoss(protein):
    hashloss =  tf.maximum(0.25*64 - tf.reduce_sum(tf.square(tf.subtract(protein,0.5)),1),0)
    return hashloss

def bitLoss(protein):
    bitloss=tf.square(tf.reduce_mean(protein,1)- 0.5)
    return bitloss

def precision(labels,predictions):
    tp=tf.reduce_sum(tf.cast(tf.cast(labels, tf.float64)*predictions[:,0] > 0,tf.int32))
    tp_fp=tf.reduce_sum(tf.cast(predictions > 0,tf.int32))
    precision=tp/tp_fp
    return precision

def recall(labels,predictions):
    tp=tf.reduce_sum(tf.cast(tf.cast(labels, tf.float64)*predictions[:,0] > 0,tf.int32))
    tp_fn=tf.reduce_sum(tf.cast(labels > 0,tf.int32))
    recall=tp/tp_fn
    return recall

def f_score(precision,recall):
    f=(2*precision*recall)/(precision+recall)
    return f


# define a function to build the models:
@tf.function
def buildModel(inputATensor,inputBTensor):
    """
    @brief Build the weights of the model
    @details The function build the model by calling it with some inputs. The aim of doing
    this is to initialize all of the model's weights. Hence, we can access the embedding
    matrix weights before we start training.
    @param inputATensor: tensor, a tensor of shape (batch_size,seq_len) which store
                         the encoded sequences of the first protein.
    @param inputBTensor: tensor, a tensor of shape (batch_size,seq_len) which store
                         the encoded sequences of the first protein.
    """
    _=model(inputATensor,inputBTensor,False) # calling the model with
    # Training mode set to False
    model.summary() # print the summary of the model

# define a function to store the embeeding matrix:
def updateEmbeddingWeights(embeddingWeightsDict,epoch):
    """
    @brief add the weights of the embedding matrix at a specific epoch to a dict object.
    @details The function keeps a record of the Embedding weights after each epochs by
    updating a dict object where keys are the epoch index and values are the
    weights of the matrix.
    @note the initial snap shot of the weights have the special key value -1.
    @param embeddingWeightsDict: dict, a dict object to store the embedding matrix weights
    @param epoch: int, the epoch index to extract and append its weights the history-dict.
    """
    weights=model.get_weights()[0]
    embeddingWeightsDict[epoch]=weights
    return embeddingWeightsDict

graphDir='DPPI_Model/'+str(sys.argv[1])
checkPoint=tf.train.Checkpoint(model=model, optimizer=optimizer)
#print(tf.train.latest_checkpoint(graphDir))
checkPoint.restore(tf.train.latest_checkpoint(graphDir))

def binaryhash(input):
    ret=np.zeros(len(input))
    for i in range(len(input)):
        #print(np.array(input[i]))
        ret[i]=1 if np.array(input[i])>0.5 else 0
    return ret

def prediction():
        for i in range(len(TestSeqA)):
            a = tf.reshape(tf.convert_to_tensor(TestSeqA[i]), [1, maxlen])
            b = tf.reshape(tf.convert_to_tensor(TestSeqB[i]), [1, maxlen])
            modelPrediction = model(a, b, False)
            #print(modelPrediction)
            aa=binaryhash(modelPrediction[0][0])
            bb=binaryhash(modelPrediction[1][0])
            dist=sum(abs(aa-bb))
            if (dist<4.1):
                print(i,": Protein",ppiset.loc[i,0],"and Protein",ppiset.loc[i,1], "have PPI.")
            else:
                print(i,": Protein",ppiset.loc[i,0],"and Protein",ppiset.loc[i,1], "don't have PPI.")

### Construct the experimental training loop:
# define the weights of the embeddingWeightsDict
embeddingWeightsDict = dict()
# build the model weights :
a=tf.convert_to_tensor(TestSeqA[0:1])
a=tf.reshape(a,[1,maxlen])
inputset=tf.keras.Input(shape=(maxlen))
buildModel(a, a)


## run the test
prediction()