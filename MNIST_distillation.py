#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Softmax
from sklearn.model_selection import KFold

print("\n # MNIST distillation  'knowledge transfer' ")

devices = tf.config.get_visible_devices()
# without metal
tf.config.set_visible_devices(devices[0])
if os.environ.get("METAL"):
    tf.config.set_visible_devices(devices[1])


class TeacherModel(Model):
    # bigger model/ensembel model
    def __init__(self, temperature: float):

        super(TeacherModel, self).__init__()
        self.temperature = temperature

        # filters: 32
        # kernel_size: 3x3
        self.conv1 = Conv2D(32, (3, 3), activation="relu")
        self.flatten = Flatten()
        
        # FNN
        # 1200 neurons 
        self.d1 = Dense(1200, activation="relu")
        self.d2 = Dense(1200, activation="relu")
        self.d3 = Dense(10)

        self.dropout = Dropout(rate=0.5)
        # multi classifcation
        self.out_sfmax = Softmax()
    
    # https://github.com/tensorflow/nmt/issues/471
    def call(self, x):
        # x input 

        x = self.conv1(x)
        x = self.flatten(x)

        x = self.d1(x)
        # x = self.dropout(x)

        x = self.d2(x)
        x = self.dropout(x)

        x = self.d3(x)
        # see equation (1) in paper
        # softmax function argument is zi/T
        # zi logits coming from dense d3 layer
        x = self.out_sfmax(x/self.temperature)
        return x


class StudentModel(Model):
    # smaller model
    def __init__(self, temperature):
        super(StudentModel, self).__init__()
        self.temperature = temperature
        self.flat = Flatten(input_shape=(28, 28))

        self.d1 = Dense(10, activation="tanh")
        self.d2 = Dense(10) 

        self.out_sfmax = Softmax()
    
    def call(self, x):
        x = self.flat(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out_sfmax(x/self.temperature)
        return x


def train_studentnet(xtrain, ytrain, xtest, ytest):
    # student net
    sf_temp_st = 6.0
    student_net = StudentModel(temperature=sf_temp_st)
    optimizer = tf.keras.optimizers.Adam()
    # from_logits explanation: http://stackoverflow.com/a/71365020
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    student_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"]) 

    # cv = 3
    # acc_scores = np.zeros(cv)
    # kf = KFold(n_splits=cv)
    # for i, (train_id, test_id) in enumerate(kf.split(xtrain)):
    #     xtrain_kf, xtest_kf = xtrain[train_id], xtrain[test_id]
    #     ytrain_kf, ytest_kf = ytrain[train_id], ytrain[test_id]
    #     student_net.fit(xtrain_kf, ytrain_kf, batch_size=32, epochs=3)
    #     ypred = student_net.predict(xtest_kf)
    #     ypred_st = np.argmax(ypred, axis=1)
    #     score = accuracy_score(ytest_kf, ypred_st)
    #     print(f"fold: {i}, score: {score}")
    #     acc_scores[i] = (score)
    
    # print(acc_scores.mean())
    student_net.fit(xtrain, ytrain, batch_size=32, epochs=3)
    ypred_st = np.argmax(student_net.predict(xtest), axis=1)
    print(f"Student Net accuracy, {accuracy_score(ytest, ypred_st)}")
    print(f"Student Net test error, {(1 - accuracy_score(ytest, ypred_st)) * xtest.shape[0]}") # student model 838 error on test dataset
    del student_net


def train_studentnet_with_distilled_knowledge(xtrain, ytrain, xtest, ytest, tensor_pred_teacher):
    print("\n # training student net with distilled knowledge transfer from teacher net")
    # train student net model with distilled knowledge of parent model(teacher net)
    from time import sleep
    sleep(3)
    sf_temp_st = 3.0
    student_net = StudentModel(temperature=sf_temp_st)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    student_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"]) 

    # cv = 3
    # acc_scores = np.zeros(cv)
    # kf = KFold(n_splits=cv)
    # for i, (train_id, test_id) in enumerate(kf.split(xtrain)):
    #     print("LOG", train_id)
    #     xtrain_kf, xtest_kf = xtrain[train_id], xtrain[test_id]
    #     ytrain_kf, ytest_kf = tensor_pred_teacher.numpy()[train_id], tensor_pred_teacher.numpy()[test_id]
    #     student_net.fit(xtrain_kf, ytrain_kf, batch_size=32, epochs=3)
    #     ypred = student_net.predict(xtest_kf)
    #     ypred_st = np.argmax(ypred, axis=1)
    #     # score = accuracy_score(ytest_kf, ypred_st)
    #     # print(f"fold: {i}, score: {score}")
    #     # acc_scores[i] = (score)
    
    # print(f"accuracy, {accuracy_score(ytest, ypred_st)}")
    # print(f"test error, {(1 - accuracy_score(ytest, ypred_st)) * xtest.shape[0]}")
    student_net.fit(xtrain, tensor_pred_teacher, batch_size=32, epochs=3)
    student_ypred = np.argmax(student_net(xtest), axis=1)
    print(f"Distilled student Net accuracy, {accuracy_score(ytest, student_ypred)}")
    print(f"Distilled student Net error, {(1- accuracy_score(ytest, student_ypred)) * xtest.shape[0]}")



if __name__ == "__main__":
    
    BATCH_SIZE = 32
    TEACHER_MODEL_FILENAME = f"teachermodel_{BATCH_SIZE}.h5py"
    YTRAIN_NPFILE = "ytrain_predictions.npy"
   
   #  MNIST dataset
    mnist_ds = tf.keras.datasets.mnist
    (xtrain, ytrain), (xtest, ytest) = mnist_ds.load_data()

    print(xtrain.shape) # (60000, 28, 28)
    print(ytrain.shape) # (60000, )
    print(xtest.shape)  # (10000, 28, 28)
    # print(ytrain[0])

    # normalize pixel value in (0, 1) range
    xtrain = xtrain/255.0
    xtest = xtest/255.0
    
    # https://stackoverflow.com/questions/47665391/keras-valueerror-input-0-is-incompatible-with-layer-conv2d-1-expected-ndim-4
    ir = xtrain.shape[1] # rows
    ic = xtrain.shape[2] # columns
    xtrain = xtrain.reshape(xtrain.shape[0], ir, ic, 1)
    xtest  = xtest.reshape(xtest.shape[0], ir, ic, 1)
    assert xtrain.shape == (60000, 28, 28, 1)
    
    # softmax temperature
    sf_temp = 3.5
    teacher_net = TeacherModel(temperature=sf_temp)

    # train teacher Net
    # older version run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    teacher_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    if os.environ.get("TRAIN"):
        print("\n training Teacher Net ")
        teacher_net.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=1, validation_split=0.2)
        teacher_net.save(TEACHER_MODEL_FILENAME)
        del teacher_net

    from tensorflow.keras.models import load_model
    teacher_net = load_model(TEACHER_MODEL_FILENAME)
    # loss: 1.5592 - accuracy: 0.8827 - val_loss: 3.0857 - val_accuracy: 0.9402
    
    # axis = 1 ( find max across row in prediction array ) return index 
    ypred = np.argmax(teacher_net(xtest), axis=1)

    # accuracy
    from sklearn.metrics import accuracy_score 
    print("\n ")
    print("Teacher Net accuracy: ", accuracy_score(ytest, ypred)*100)
    # errors
    test_error = int((1 - accuracy_score(ytest, ypred)) * ytest.shape[0]) # 575 test error 
    print(f"Teacher Net test_error: {test_error}")
    
    # due to GPU memory exhaustion error (8 GB RAM)
    # take predictions from first 40000 half and then the remaning
    LEN_IMAGES = 40000
    # print(xtrain.shape)
    # print(xtrain[:LEN_IMAGES].shape)
    # if not os.path.isfile(YTRAIN_NPFILE):
    
    ytrain_pred_one = teacher_net(xtrain[:LEN_IMAGES]).numpy()
    ytrain_pred_two = teacher_net(xtrain[LEN_IMAGES:]).numpy()
    ytrain_pred = (np.concatenate([ytrain_pred_one, ytrain_pred_two], axis=0))
    
    tensor_ytrain = tf.convert_to_tensor(ytrain_pred)
    # print("LOG", tensor_ytrain.numpy().shape)
    # print("LOG", ytrain.shape)
    # print("LOG", tensor_ytrain.numpy()[[0, 1]])

   # with open(YTRAIN_NPFILE, "wb") as f:
        #     np.save(f, ytrain_pred)
    
    # with open(YTRAIN_NPFILE, "rb") as f:
        # ytrain_pred = np.load(f, allow_pickle=True)


    # student net
    train_studentnet(xtrain, ytrain, xtest, ytest) 
    train_studentnet_with_distilled_knowledge(xtrain, ytrain, xtest, ytest, tensor_ytrain) 





















