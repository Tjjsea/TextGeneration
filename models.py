#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class MTA():
    def __init__(self,flag):
        pass

class Seq2Seq():
    def __init__(self,flag):
        self.encoder_input=tf.placeholder(tf.int32,[None,None],name="encoder_input")
        self.encoder_input_length=tf.placeholder(tf.int32,[None],name="encoder_input_length")
        self.decoder_target=tf.placeholder(tf.int32,[None,None],name="decoder_target")
        self.decoder_target_length=tf.placeholder(tf.int32,[None],name="decoder_target_length")
        self.keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.l2loss=tf.constant(0.0)

        #flag: vocab_size, embedding_dim, en_hid(hidden size of encoder rnn)

        with tf.variable_scope("encoder"):
            encoder_embedding=tf.Variable(tf.random_uniform([flag.vocab_size,flag.embedding_dim],-1.0,1.0),name='encoder_embedding')
            encoder_embedded=tf.nn.embedding_lookup(encoder_embedding,self.encoder_input)

            enfw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(flag.en_hid))
            enbw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(flag.en_hid))
            outputs,states=tf.nn.bidirectional_dynamic_rnn(enfw,enbw,dtype=tf.float32)
            self.encoder_out=tf.concat(outputs,-1)

        pass