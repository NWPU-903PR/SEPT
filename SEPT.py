# -*- coding: utf-8 -*-
"""

"""

from keras.layers import Input, Dense, Dropout, Flatten, Lambda   
from keras.layers import Conv1D, MaxPooling1D,concatenate,Activation
from keras.layers import  BatchNormalization, CuDNNLSTM, Bidirectional
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import  l2
from keras.engine import Layer

import os
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import metrics

import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)
# Helper functions
K.set_image_dim_ordering('tf')

batch_size = 64
nb_epoch = 80
en_long, pr_long = 3000,2000
nb_filters = 300
nb_pool = 20
nb_conv = 40
LSTM_out_dim = 50
domain_dense = 50
class_dense = 100
REVERSE_RATE = 1
learning_rate = 0.001

_TRAIN = K.variable(1, dtype='uint8')
filepath=r'E:\Users\fjing\share\projects\epi0415\data\six_cell/'

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]

def batch_gen(batches, id_array, data_e, data_p, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_start = int(batch_start)
        batch_end = int(batch_end)
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data_e[batch_ids],data_p[batch_ids], labels[batch_ids]
        else:
            yield data_e[batch_ids],data_p[batch_ids]
        np.random.shuffle(id_array)

class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD(lr =learning_rate)

    def _build_feature_extractor(self, enhancer_input,promoter_input):
        enhancer_net = Conv1D(nb_filters, nb_conv,
                            padding='valid',
                            W_regularizer = l2(1e-5))(enhancer_input)
        enhancer_net = Activation("relu")(enhancer_net)        
        enhancer_net = MaxPooling1D(nb_pool, nb_pool)(enhancer_net)
        
        enhancer_net = Conv1D(nb_filters, nb_conv,
                            W_regularizer = l2(1e-5))(enhancer_net)
        enhancer_net = Activation("relu")(enhancer_net)        
        enhancer_net = MaxPooling1D(nb_pool, nb_pool)(enhancer_net)
        
        promoter_net = Conv1D(nb_filters, nb_conv,
                            padding='valid',
                            W_regularizer = l2(1e-5))(promoter_input)
        promoter_net= Activation("relu")(promoter_net)        
        promoter_net = MaxPooling1D(nb_pool, nb_pool)(promoter_net)    
        
        promoter_net = Conv1D(nb_filters, nb_conv,
                            W_regularizer = l2(1e-5))(promoter_net)
        promoter_net= Activation("relu")(promoter_net)        
        promoter_net = MaxPooling1D(nb_pool, nb_pool)(promoter_net) 
        
        merge_layer = concatenate([enhancer_net,promoter_net],
                                  axis = 1,name ='merge_layer')  
        batch_net_1 = BatchNormalization()(merge_layer)
        drop_net_1  = Dropout(0.25)(batch_net_1 )

        biLSTM_layer  = Bidirectional(CuDNNLSTM(units = LSTM_out_dim,
										return_sequences = True,name = 'biLSTM_layer'))(drop_net_1)
        batch_layer_2 = BatchNormalization()(biLSTM_layer)
        drop_layer_2  = Dropout(0.5)(batch_layer_2 )  
        flatten_layer = Flatten(name = 'invariant_features')(drop_layer_2)

        self.domain_invariant_features = flatten_layer
        return flatten_layer

    def _build_classifier(self, model_input):
        net = Dense(class_dense,W_regularizer = l2(1e-6))(model_input)
        net = BatchNormalization()(net)
        net = Activation("relu")(net)
        net = Dropout(0.5)(net)
        net = Dense(1, activation='sigmoid',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, enhancer_input,promoter_input):
        net = self._build_feature_extractor(enhancer_input,promoter_input)
        net = self._build_classifier(net)
        model = Model(inputs=[enhancer_input,promoter_input], outputs=net)
        model.compile(loss={'classifier_output': 'binary_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, enhancer_input,promoter_input):                        #1
        net = self._build_feature_extractor(enhancer_input,promoter_input)
        Flip = GradientReversal(REVERSE_RATE)
        self.grl = Flip
        branch = self.grl(net)
        branch = Dense(domain_dense,W_regularizer = l2(1e-6))(branch)
        branch = BatchNormalization()(branch)
        branch = Activation("relu")(branch)        
        branch = Dropout(0.5)(branch)
        branch = Dense(1, activation='sigmoid', name='domain_output')(branch)
        net = Lambda(lambda x: K.switch(K.learning_phase(),
                     x[:int(batch_size / 2), :], x),
                     output_shape=lambda x: ((batch_size / 2,) +
                     x[1:]) if _TRAIN else x[0:])(net)

        net = self._build_classifier(net)
        model = Model(inputs=[enhancer_input,promoter_input], outputs=[branch, net])

        model.compile(loss={'classifier_output': 'binary_crossentropy',
                      'domain_output': 'binary_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

def combine_npy(filepath,test_split=0.5):
    #test_cell = 'K562' 
    
    X_e = np.load(filepath + "/train6enhancer.npy")
    X_e_test = np.load(filepath + "/enhancer_of_pairs.npy")
    
    X_p = np.load(filepath + "/train6promoter.npy")
    X_p_test = np.load(filepath + "/promoter_of_pairs.npy") 
    
    y = np.load(filepath + "/train6label.npy")
    y_test = np.load(filepath + "/label.npy") 
    X_e_src_train,X_e_src_test, X_p_src_train,X_p_src_test,  y_src_train,y_src_test = train_test_split(X_e,X_p, y, test_size=0.15, random_state=42)      
    X_e_tgt_train,X_e_tgt_test, X_p_tgt_train,X_p_tgt_test, y_tgt_train,y_tgt_test  = train_test_split(X_e_test,X_p_test, y_test, test_size=0.5, random_state=42)  
    return X_e_src_train,X_p_src_train,y_src_train, X_e_src_test,X_p_src_test,y_src_test,X_e_tgt_train,X_p_tgt_train,y_tgt_train,  X_e_tgt_test,X_p_tgt_test,y_tgt_test, test_cell

#cellnames =  ['K562', 'GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'NHEK','HMEC']
X_e_src_train,X_p_src_train,y_src_train, X_e_src_test,X_p_src_test,y_src_test,X_e_tgt_train,X_p_tgt_train,y_tgt_train,  X_e_tgt_test,X_p_tgt_test,y_tgt_test, test_cell    = combine_npy(filepath)  

domain_labels = np.vstack([np.tile([1], [int(batch_size / 2), 1]),
                           np.tile([0], [int(batch_size / 2), 1])])

enhancer_input = Input(shape=(en_long,4), name='e_input')
promoter_input = Input(shape=(pr_long,4), name='p_input')
builder = DANNBuilder()
dann_model = builder.build_dann_model(enhancer_input,promoter_input)

src_index_arr = np.arange(X_e_src_train.shape[0])
target_index_arr = np.arange(X_e_tgt_train.shape[0])

batches_per_epoch = len(X_e_src_train) // batch_size
num_steps = nb_epoch * batches_per_epoch

print('Training DANN model：')
for i in range(1,nb_epoch+1):
    batches        = make_batches(X_e_src_train.shape[0], batch_size // 2)
    target_batches = make_batches(X_e_tgt_train.shape[0], batch_size // 2)
    src_gen    = batch_gen(batches,        src_index_arr,   X_e_src_train,X_p_src_train,y_src_train)
    target_gen = batch_gen(target_batches, target_index_arr,X_e_tgt_train,X_p_tgt_train, None)

    epoch_loss = []
    print('Epoch {}/{}'.format(i,nb_epoch))
    
    builder.grl.l = 1
    builder.opt.lr = learning_rate
    
    for (xb_e,xb_p, yb) in src_gen:
        if xb_e.shape[0] != batch_size / 2:
            continue
        try:
            xt_e,xt_p = target_gen.__next__()
        except:
            # Regeneration
            target_gen = batch_gen(target_batches, target_index_arr,X_e_tgt_train,X_p_tgt_train, None)
        # Concatenate source and target batch
        xb_e = np.vstack([xb_e, xt_e])
        xb_p = np.vstack([xb_p, xt_p])
        if xb_e.shape[0] == batch_size:
            dann_history = dann_model.train_on_batch([xb_e,xb_p],
                                                {'classifier_output': yb,
                                                'domain_output': domain_labels}
                                                )

print('Evaluating target samples on DANN model')
y_pred = dann_model.predict([X_e_tgt_test, X_p_tgt_test])
y_pred = y_pred[1]
auc = metrics.roc_auc_score(y_tgt_test, y_pred)
print ('AUC on test set:\n',auc)









