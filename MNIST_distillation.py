#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Softmax, InputLayer
from tensorflow.keras import Input
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 

print("\n # MNIST distillation  'knowledge transfer' ")

BATCH_SIZE = 32
TEACHER_MODEL_FILENAME = f"teachermodel_{BATCH_SIZE}.h5py"
TNET_WEIGHTS_MODEL_FILENAME = "tnet_logits.h5"
YTRAIN_NPFILE = "ytrain_predictions.npy"

devices = tf.config.get_visible_devices()
# without metal
tf.config.set_visible_devices(devices[0])
if os.environ.get("METAL"):
    tf.config.set_visible_devices(devices[1])


class TeacherModel(Model):
    # bigger model/ensembel model
    def __init__(self, temperature: float, out_logits=False):

        super(TeacherModel, self).__init__()
        self.temperature = temperature
        self.out_logits = out_logits

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
        if self.out_logits:
            return x
        x = self.out_sfmax(x/self.temperature)
        return x



class StudentModel(Model):
    # smaller model
    def __init__(self):
        super(StudentModel, self).__init__()
        self.flat = Flatten(input_shape=(28, 28))
        self.d1 = Dense(10, activation="tanh")
        self.d2 = Dense(10) 

    
    def call(self, x):
        # return logits
        x = self.flat(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

    def build(self, input_shape):
        print("Ok- building student net model")
        return super(StudentModel, self).build(input_shape)



# student net with KD loss function
class StudnetNet_KDLoss(Model):
    def __init__(self, teachernet, studentnet, temperature, alpha, beta):
        super(StudnetNet_KDLoss, self).__init__()
        self.studentnet = studentnet
        self.teachernet = teachernet 
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")
    
    def train_step(self, x):
        print("\nOk- training StudnetNet_KDLoss model")
        teacher_logits = self.teachernet(x[0])
        # print("Ok-", self.studentnet.trainable_variables)
        with tf.GradientTape() as tape:
            # forward passj
            student_logits = self.studentnet(x[0], training=True)
            # compute my own KD loss
            loss = self.add_kd_loss(student_logits, teacher_logits, x[1])
        
        # compute gradients
        gradients = tape.gradient(loss, self.studentnet.trainable_variables)
        # read section 2 / "Distillation" end paragraph
        # Since the magnitudes of the gradients produced by the soft targets scale as 1/T^2
        # it is important to multiply them by T^2 when using both hard and soft targets.
        # This ensures that the relative contributions of the hard and soft targets remain roughly unchanged 
        # if the temperature used for distillation is changed while experimenting with meta-parameters.
        gradients = [gradient * (self.temperature**2) for gradient in gradients]

        # update weights
        self.optimizer.apply_gradients(zip(gradients, self.studentnet.trainable_variables))
        
        # metrics
        self.train_loss.update_state(loss)
        self.train_accuracy(x[1], tf.nn.softmax(student_logits))
        result_loss = self.train_loss.result()
        result_accuracy = self.train_accuracy.result()
        self.train_loss.reset_states(); self.train_accuracy.reset_states()
        metric = {"loss": result_loss, "accuracy": result_accuracy}
        return metric


    def test_step(self, x):
        teacher_logits = self.teachernet(x[0])
        student_logits = self.studentnet(x[0], training=False)
        loss = self.add_kd_loss(student_logits, teacher_logits, x[1])
        
        # metrics
        self.valid_loss.update_state(loss)
        self.valid_accuracy(x[1], tf.nn.softmax(student_logits))
        result_loss = self.valid_loss.result()
        result_accuracy = self.valid_accuracy.result()
        self.valid_loss.reset_states(); self.valid_accuracy.reset_states()
        metric = {"loss": result_loss, "accuracy": result_accuracy}
        return metric

    def add_kd_loss(self, student_logits, teacher_logits, ytrain_vals):
        # compute knowlegde distillation loss
        # https://discuss.pytorch.org/t/knowledge-distillation-what-loss/196494
        # https://techtalkwithsriks.medium.com/building-small-language-models-using-knowledge-distillation-kd-6825ce2f6d24
        # KD loss = KL loss + Cross Entropy loss
        # KD = KL divergence loss + Cross entropy loss
        # y: true labels (generator)
        teacher_prob = tf.nn.softmax(teacher_logits/self.temperature)
        # https://stackoverflow.com/a/68617676
        kd_loss = tf.keras.losses.categorical_crossentropy(teacher_prob, student_logits/self.temperature, from_logits=True)
        sce_loss = tf.keras.losses.sparse_categorical_crossentropy(ytrain_vals, student_logits, from_logits=True)
        loss = (self.alpha * kd_loss) + (self.beta * sce_loss)
        return (loss) / (self.alpha + self.beta)


def student_KDLoss(xtrain, ytrain, xtest, ytest):
    snetmod = StudentModel()
    # NOTE: it is important to build the model first
    # if not then self.studentnet.trainable_variables will return None
    # in StudnetNet_KDLoss class
    snetmod.build(xtrain.shape)
    # print(snetmod.summary())

    # training Teacher Net 
    tnetmod = TeacherModel(temperature=1.0, out_logits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    tnetmod.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    if os.environ.get("TRAIN"):
        print("\n Ok- training Teacher Net")
        tnetmod.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2)
        tnetmod.save_weights(TNET_WEIGHTS_MODEL_FILENAME, overwrite=True)
    
    tnetmod.build(xtrain.shape)
    tnetmod.load_weights(TNET_WEIGHTS_MODEL_FILENAME)

    # training student net model without DK loss
    print("Ok- training student net model without DK")
    train_studentnet(xtrain, ytrain, xtest, ytest)

    # training student net with distillation knowledge
    if os.environ.get("TRAIN"):
        print("\n Ok- training student net with knowledge distillation")
        snet_kdloss = StudnetNet_KDLoss(tnetmod, snetmod, temperature=5.0, alpha=0.9, beta=0.1)
        snet_kdloss.compile(optimizer=optimizer)
        snet_kdloss.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=5, validation_split=0.2)
        snet_kdloss.studentnet.save_weights("snet_kdloss_studentnet_weights.h5")

    # compute metrics
    snetmod.load_weights("snet_kdloss_studentnet_weights.h5")
    ypred_st = np.argmax(snetmod.predict(xtest), axis=1)
    print(f"Student Net after DK accuracy, {accuracy_score(ytest, ypred_st)}")
    print(f"Student Net after DK test error, {int((1 - accuracy_score(ytest, ypred_st)) * ytest.shape[0])}") 



def train_studentnet(xtrain, ytrain, xtest, ytest):
    # student net
    student_net = StudentModel()
    optimizer = tf.keras.optimizers.Adam()
    # from_logits explanation: http://stackoverflow.com/a/71365020
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    student_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"]) 
    student_net.fit(xtrain, ytrain, batch_size=32, epochs=5, validation_split=0.2)
    student_net.save_weights("student_net_weights.h5")
    ypred_st = np.argmax(student_net.predict(xtest), axis=1)
    print(f"Student Net accuracy, {accuracy_score(ytest, ypred_st)}")
    print(f"Student Net test error, {int((1 - accuracy_score(ytest, ypred_st)) * ytest.shape[0])}") 


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
    student_net.fit(xtrain, tensor_pred_teacher, batch_size=32, epochs=3)
    student_ypred = np.argmax(student_net(xtest), axis=1)
    print(f"Distilled student Net accuracy, {accuracy_score(ytest, student_ypred)}")
    print(f"Distilled student Net error, {int((1- accuracy_score(ytest, student_ypred)) * ytest.shape[0])}")



if __name__ == "__main__":
    
  
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
    # optimizer = tf.keras.optimizers.Adam()
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # teacher_net.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    # if os.environ.get("TRAIN"):
    #     print("\n training Teacher Net ")
    #     teacher_net.fit(xtrain, ytrain, batch_size=BATCH_SIZE, epochs=1, validation_split=0.2)
    #     teacher_net.save(TEACHER_MODEL_FILENAME)
    #     del teacher_net

    # from tensorflow.keras.models import load_model
    # teacher_net = load_model(TEACHER_MODEL_FILENAME)
    # # loss: 1.5592 - accuracy: 0.8827 - val_loss: 3.0857 - val_accuracy: 0.9402
    
    # # axis = 1 ( find max across row in prediction array ) return index 
    # ypred = np.argmax(teacher_net(xtest), axis=1)

    # # accuracy
    # print("\n ")
    # print("Teacher Net accuracy: ", accuracy_score(ytest, ypred)*100)
    # # errors
    # test_error = int((1 - accuracy_score(ytest, ypred)) * ytest.shape[0]) # 575 test error 
    # print(f"Teacher Net test_error: {test_error}")
    
    # due to GPU memory exhaustion error (8 GB RAM)
    # take predictions from first 40000 half and then the remaning
    LEN_IMAGES = 40000
    # print(xtrain.shape)
    # print(xtrain[:LEN_IMAGES].shape)
    # if not os.path.isfile(YTRAIN_NPFILE):
    
    # ytrain_pred_one = teacher_net(xtrain[:LEN_IMAGES]).numpy()
    # ytrain_pred_two = teacher_net(xtrain[LEN_IMAGES:]).numpy()
    # ytrain_pred = (np.concatenate([ytrain_pred_one, ytrain_pred_two], axis=0))
    # tensor_ytrain = tf.convert_to_tensor(ytrain_pred)

    # # print("LOG", tensor_ytrain.numpy().shape)
    # print("LOG", ytrain.shape)
    # print("LOG", tensor_ytrain.numpy()[[0, 1]])

   # with open(YTRAIN_NPFILE, "wb") as f:
        #     np.save(f, ytrain_pred)
    
    # with open(YTRAIN_NPFILE, "rb") as f:
        # ytrain_pred = np.load(f, allow_pickle=True)


    # student net
    # train_studentnet(xtrain, ytrain, xtest, ytest) 
    # train_studentnet_with_distilled_knowledge(xtrain, ytrain, xtest, ytest, tensor_ytrain) 



    student_KDLoss(xtrain, ytrain, xtest, ytest)
















