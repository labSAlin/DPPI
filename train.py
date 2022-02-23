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
    print("Usage: python train.py [dataset]")
    exit(1)


train_batchsize=60
test_batchsize=400


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

# read the Pos/Neg data


negset=pd.read_table(data_dir+'NegativeEdges.txt',sep='\t',header=None)
posset=pd.read_table(data_dir+'PositiveEdges.txt',sep='\t',header=None)

data1=[]
data2=[]
label=[]
for i in range(len(negset)):
    data1.append(enseq[negset.loc[i,0]])
    data2.append(enseq[negset.loc[i,1]])

label=label+[0]*len(negset)

for i in range(len(posset)):
    data1.append(enseq[posset.loc[i,0]])
    data2.append(enseq[posset.loc[i,1]])


label=label+[1]*len(posset)

# Make the train dataset and the test dataset
total=len(label)

idxTrain = random.sample(range(total),int(total*0.9))
idxTest = list(set(range(total))-set(idxTrain))

# Make BA samples in the train dataset
TrainSeqA=[]
TrainSeqB=[]
TrainLabel=[]
for i in idxTrain:
    TrainSeqA.append(data1[i])
    TrainSeqB.append(data2[i])
    TrainSeqA.append(data2[i])
    TrainSeqB.append(data1[i])
    TrainLabel.append(label[i])
    TrainLabel.append(label[i])

# Make test data set
TestSeqA=[]
TestSeqB=[]
TestLabel=[]
for i in idxTest:
    TestSeqA.append(data1[i])
    TestSeqB.append(data2[i])
    TestLabel.append(label[i])


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
#trainAcuracy=tf.keras.metrics.Accuracy(name="TrainAccuracy")
#trainAUC=tf.keras.metrics.AUC(name="TrainAUC")
#trainRecall=tf.keras.metrics.Recall(name="TrainRecall")
#trainPrecision=tf.keras.metrics.Precision(name="TrainPercision")
trainPrecision=tf.keras.metrics.Mean(name="TrainPercision")
trainRecall=tf.keras.metrics.Mean(name="TrainRecall")
trainF=tf.keras.metrics.Mean(name="TrainF")
# test metrics:
testLoss=tf.keras.metrics.Mean(name="testLoss")
testBasicloss=tf.keras.metrics.Mean(name="testBasicloss")
testHashloss=tf.keras.metrics.Mean(name="testHashloss")
testBitloss=tf.keras.metrics.Mean(name="testBitloss")
#testAcuracy=tf.keras.metrics.Accuracy(name="TestAccuracy")
#testAUC=tf.keras.metrics.AUC(name="TestAUC")
#testRecall=tf.keras.metrics.Recall(name="TestRecall")
#testPrecision=tf.keras.metrics.Precision(name="TestPercision")
testRecall=tf.keras.metrics.Mean(name="TestRecall")
testPrecision=tf.keras.metrics.Mean(name="TestPercision")
testF=tf.keras.metrics.Mean(name="TestF")
## define the train and evaluation loop:
# define the input signature:
InputSignature=[
         tf.TensorSpec(shape=(None,len(TrainSeqA[0])),
                       dtype=tf.int32),
         tf.TensorSpec(shape=(None,len(TrainSeqA[0])),
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
ckpt_manager = tf.train.CheckpointManager(checkPoint, graphDir,
                                          max_to_keep=100)


# define the training step function
@tf.function(input_signature=InputSignature)
def trainStep(inputATensor,inputBTensor,label):
    """
    @brief train the model on a batch of data
    @details The function train the model using one batch of the input data.
             Training is carried out using GradientTapes.
    @param inputATensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain
    the tokenized sequences of the first protein.
    @param inputBTensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain
    the tokenized sequences of the second protein.
    @param label: tensor, a rank one tensor which store for each pair of input sequences whether
    they interact of not.
    @note: During Accuracy calculations the model predictions are casted
    as binary values, i.e. if the predictions are 0.5 AND above they are
    casted to One otherwise to zero.
    """
    with tf.GradientTape() as tape:

        modelPrediction=model(inputATensor,inputBTensor,True)
        dist = tf.reduce_sum(tf.abs(tf.subtract(modelPrediction[0], modelPrediction[1])),1)
        basic_Loss=basicLoss(label,dist,train_batchsize)
        hash_loss1=hashLoss(modelPrediction[0])
        hash_loss2=hashLoss(modelPrediction[1])
        hash_loss=tf.reduce_mean(tf.add(hash_loss1,hash_loss2))/32
        bitloss1=bitLoss(modelPrediction[0])
        bitloss2=bitLoss(modelPrediction[1])
        bitloss=tf.reduce_mean(tf.add(bitloss1,bitloss2))
        loss=basic_Loss+hash_loss+bitloss
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    trainLoss(loss)  #计算loss的加权平均值
    trainBasicloss(basic_Loss)
    trainHashloss(hash_loss)
    trainBitloss(bitloss)
    dist=tf.reshape(dist,[-1,1])
    threshold_dist=tf.cast(dist<=4,tf.float64)
    train_precision=precision(label,threshold_dist)
    trainPrecision(train_precision)
    train_recall=recall(label,threshold_dist)
    trainRecall(train_recall)
    train_f=f_score(train_precision,train_recall)
    trainF(train_f)
    return dist, label,modelPrediction[0], modelPrediction[1]

# define the Test function:
@tf.function(input_signature=InputSignature)
def testStep(inputATensor,inputBTensor,label):
    """
    @brief Test the model performance on a batch of input data
    @details The function test the model using one batch of the input validation data.
    @param inputATensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain
    the tokenized sequences of the first protein.
    @param inputBTensor: tensor, a rank two tensor of shape (batch_size, seq_len) which contain
    the tokenized sequences of the second protein.
    @param label: tensor, a rank one tensor which store for each pair of input sequences whether
    they interact of not.
    @note: During Accuracy calculations the model predictions are casted
    as binary values, i.e. if the predictions are 0.5 AND above they are
    casted to One otherwise to zero.
    """
    modelPrediction=model(inputATensor,inputBTensor,False)
    dist = tf.reduce_sum(tf.abs(tf.subtract(modelPrediction[0], modelPrediction[1])),1)
    basic_Loss=basicLoss(label,dist,test_batchsize)
    hash_loss1=hashLoss(modelPrediction[0])
    hash_loss2=hashLoss(modelPrediction[1])
    hash_loss=tf.reduce_mean(tf.add(hash_loss1,hash_loss2))
    bitloss1=bitLoss(modelPrediction[0])
    bitloss2=bitLoss(modelPrediction[1])
    bitloss=tf.reduce_mean(tf.add(bitloss1,bitloss2))
    loss=basic_Loss+hash_loss+bitloss
    testLoss(loss)  #计算loss的加权平均值
    testBasicloss(basic_Loss)
    testHashloss(hash_loss)
    testBitloss(bitloss)
    dist=tf.reshape(dist,[-1,1])
    threshold_dist=tf.cast(dist<=8.1,tf.float64)
    test_precision=precision(label,threshold_dist)
    testPrecision(test_precision)
    test_recall=recall(label,threshold_dist)
    testRecall(test_recall)
    test_f=f_score(test_precision,test_recall)
    testF(test_f)
    return dist,label


# define a function to train the model:
def trainEpoch(numEpoch,  embeddingWeightsDict):
    """
    @brief train and evaluate the models for a specific number of epochs.
    @details The function trains the model for a specific number of epochs defined by
    the variable numEpoch, after each epoch it update a record of the model
    performance on the training and test data-sets and it stores the weights of the
    embedding layer.
    @param numEpoch:int, it is the number of training epochs.
    @param outputPath: int, it is the path to store the model performance logs and the embedding layer weights.
    @param embeddingWeightsDict: dict, it is a dict object to store the embedding layer
        weights during the training.
    """
    # a history object to store the data
    history = dict()
    for epoch in range(numEpoch):
        # reset the training metrics:
        trainLoss.reset_states()
        trainBasicloss.reset_states()
        trainHashloss.reset_states()
        trainBitloss.reset_states()
        # trainAcuracy.reset_states()
        # trainAUC.reset_states()
        trainRecall.reset_states()
        trainPrecision.reset_states()
        trainF.reset_states()
        # define metric variables to store the results:
        trainloss = 0
        for (batch, (inputTensorA, inputTensorB, labels)) in enumerate(trainDataset):
            # print("batch:",batch,"shape",inputTensorA.shape,inputTensorB.shape,labels.shape)
            if (inputTensorA.shape[0] != train_batchsize):
                continue
            train_out = trainStep(inputTensorA, inputTensorB, labels)

            if batch % 100 == 0:  # print the model state to the console every 5 batch
                print("\nEpoch: {}, Training Batch {} State: ".format(epoch, batch))
                print("loss: {} \t Recall: {} \t Precision: {}\t F-score:{} \t".format(
                    trainLoss.result(), trainRecall.result(),
                    trainPrecision.result(), trainF.result()))
                print("basicloss: {} \t ".format(trainBasicloss.result()))
                print("hashloss: {} \t ".format(trainHashloss.result()))
                print("bitloss: {} \t ".format(trainBitloss.result()))
                print("dist:", train_out[0][0:10], "\nlabel:", train_out[1][0:10])
        trainloss = trainLoss.result().numpy()
        trainrecall = trainRecall.result().numpy()
        trainprecision = trainPrecision.result().numpy()
        trainf = trainF.result().numpy()

        if epoch % 100 == 0:
            ckpt_manager.save()

### Construct the experimental training loop:
# define the weights of the embeddingWeightsDict
embeddingWeightsDict = dict()
# build the model weights :
a=tf.convert_to_tensor(TrainSeqA[0:10])
a=tf.reshape(a,[10,maxlen])
inputset=tf.keras.Input(shape=(maxlen))
buildModel(a, a)

## Construct the input Data sets:
print("Convert to train set")
numberOfTrainingExamples = len(TrainSeqA)
trainDataset = tf.data.Dataset.from_tensor_slices(
    (
        TrainSeqA,
        TrainSeqB,
        TrainLabel)
).shuffle(1000).batch(train_batchsize)
print("Convert to test set")
testDataset = tf.data.Dataset.from_tensor_slices(
    (TestSeqA, TestSeqB,
     TestLabel)).shuffle(1000).batch(test_batchsize)

## Start the training loop
trainEpoch(1,  embeddingWeightsDict)
