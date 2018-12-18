# """Tensorflow high-level API example (estimator, contrib.learner)
# Train cifar10 with VGG16
# Before running, generate cifar10 tfrecords according to the official cifar10 estimator code:
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
# and save them at ./data/cifar-10-data
# Also, cifar10.py should be downloaded from the url as well.
# pure estimator running by: python tf_estimator_vgg_cifar10.py --api-lv=0
# with tf.contrib.learner: python tf_estimator_vgg_cifar10.py --api-lv=1
# """
# # py2 - py3 compatibility settings
# from __future__ import absolute_import, division, print_function, unicode_literals
# from six.moves import xrange
# # build-in libraries
# import os
# import pdb
# import argparse
# import functools
# # pre-installed libraries
# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# # local files
# import cifar10

# tf.logging.set_verbosity(tf.logging.INFO)
# slim = tf.contrib.slim

# class VGG16:

#     def __init__(self, is_training, n_classes):
#         self.is_training = is_training
#         self.n_classes = n_classes
#         self.weight_decay = 0.0005

#     def forward(self, features):
#         with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
#                 weights_regularizer=slim.l2_regularizer(self.weight_decay)):

#             net = slim.repeat(features, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#             net = slim.max_pool2d(net, [2, 2], scope='pool1')
#             net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#             net = slim.max_pool2d(net, [2, 2], scope='pool2')
#             net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#             net = slim.max_pool2d(net, [2, 2], scope='pool3')
#             net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#             net = slim.max_pool2d(net, [2, 2], scope='pool4')
#             net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#             net = slim.max_pool2d(net, [2, 2], scope='pool5')

#             logits = slim.conv2d(net, self.n_classes, [1, 1], scope='fc6')
#             logits = tf.squeeze(logits, [1, 2], name='fc6/squeezed')
#             predictions = tf.argmax(logits, axis=-1)

#         return logits, predictions


# def get_model_fn(features, labels, mode, params):
#     """Get model_fn for tf.estimator.Estimator
#     tf.estimator.EstimatorSpec args (basic):
#         - mode
#         - predictions
#         - loss
#         - train_op
#         - eval_metric_ops
#     Args:
#         features: default features  (batch, 32, 32, 3)
#         labels: default labels      (batch)
#         mode: default tf.estimator.ModeKeys
#         params: default HParams object
#     Return:
#         tf.estimator.EstimatorSpec
#     """
#     # get custom model
#     is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#     model = VGG16(is_training, params.n_classes)
#     logits, predictions = model.forward(features)
#     if mode != tf.estimator.ModeKeys.PREDICT:
#         # loss
#         loss = tf.losses.sparse_softmax_cross_entropy(
#             labels=tf.cast(labels, tf.int32),
#             logits=logits
#         )

#         # train_op
#         decay_steps = params.iters_ep * 10
#         lr = tf.train.exponential_decay(
#             params.lr,
#             tf.train.get_or_create_global_step(),
#             decay_steps,
#             0.9, staircase=True
#         )
#         train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=tf.train.get_or_create_global_step())

#         # eval_metric_ops
#         eval_metric_ops = {
#             'accuracy': tf.metrics.accuracy(
#                 labels=labels,
#                 predictions=predictions,
#                 name='accuracy'
#             )
#         }
#     else:
#         loss = None
#         train_op = None
#         eval_metric_ops=None

#     return tf.estimator.EstimatorSpec(
#         mode=mode,
#         predictions=predictions,
#         loss=loss,
#         train_op=train_op,
#         eval_metric_ops=eval_metric_ops)

# def get_experiment_fn(run_config, hparams):
#     """Get experiment_fn for tf.contrib.learn.learn_runner
#     tf.contrib.learn.Experiment args (basic):
#         - tf.estimator.Estimator
#         - train_input_fn/eval_input_fn
#         - TBD
#     Args:
#         run_config: default RunConfig object
#         hparams: default HParams object
#     Return:
#         tf.contrib.learn.Experiment
#     """
#     # get estimator from model_fn
#     estimator = tf.estimator.Estimator(
#         model_fn=get_model_fn,
#         params=hparams,
#         config=run_config
#     )

#     # setup data loaders
#     dataset_train = cifar10.Cifar10DataSet(hparams.data_dir, 'train')
#     train_input_fn = functools.partial(dataset_train.make_batch, batch_size=hparams.bsize)

#     dataset_eval = cifar10.Cifar10DataSet(hparams.data_dir, 'eval')
#     eval_input_fn = functools.partial(dataset_eval.make_batch, batch_size=hparams.bsize_eval)

#     return tf.contrib.learn.Experiment(
#         estimator=estimator,
#         train_input_fn=train_input_fn,
#         eval_input_fn=eval_input_fn,
#         train_steps=hparams.train_steps,
#         eval_steps=cifar10.Cifar10DataSet.num_examples_per_epoch('eval') // hparams.bsize_eval)

# def get_configs(args):

#     hparams = tf.contrib.training.HParams(
#         train_steps=args.ep * cifar10.Cifar10DataSet.num_examples_per_epoch() // args.bsize,
#         iters_ep=cifar10.Cifar10DataSet.num_examples_per_epoch() // args.bsize,
#         n_classes=10,
#         **vars(args)
#     )

#     session_config = tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=args.dev_place,
#         gpu_options=tf.GPUOptions(
#             force_gpu_compatible=True,
#             allow_growth=True)
#     )

#     run_config = tf.contrib.learn.RunConfig(
#         model_dir=args.model_dir,
#         tf_random_seed=args.rseed,
#         save_checkpoints_steps=hparams.iters_ep,
#         log_step_count_steps=hparams.iters_ep, # only log every epoch
#         session_config=session_config
#     )

#     return hparams, run_config

# def main(args):
#     """Main function"""
#     # get settings
#     hparams, run_config = get_configs(args)

#     if args.api_lv == 0:
#         # get estimator
#         estimator = tf.estimator.Estimator(
#             model_fn=get_model_fn,
#             params=hparams,
#             config=run_config
#         )
#         # setup data loaders
#         dataset_train = cifar10.Cifar10DataSet(hparams.data_dir, 'train')
#         train_input_fn = functools.partial(dataset_train.make_batch, batch_size=hparams.bsize)

#         dataset_eval = cifar10.Cifar10DataSet(hparams.data_dir, 'eval')
#         eval_input_fn = functools.partial(dataset_eval.make_batch, batch_size=hparams.bsize_eval)

#         for ep in xrange(hparams.ep):
#             estimator.train(input_fn=train_input_fn, steps=hparams.iters_ep)
#             estimator.evaluate(input_fn=eval_input_fn, steps=cifar10.Cifar10DataSet.num_examples_per_epoch('eval') // hparams.bsize_eval)

#     elif args.api_lv == 1:
#         # get experiment
#         tf.contrib.learn.learn_runner.run(
#             experiment_fn=get_experiment_fn,
#             schedule='train_and_evaluate',
#             run_config=run_config,
#             hparams=hparams
#         )
#     print ('All Done')
#     # done

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # API-Level
#     parser.add_argument('--api-lv',
#         type=int,
#         default=0,
#         help='0: pure estimator, 2: contrib.learner.')
#     # Hyperparameters
#     parser.add_argument('--rseed',
#         type=int,
#         default=420,
#         help='random seed.')
#     parser.add_argument('--lr',
#         type=float,
#         default=2e-2,
#         help='learning rate.')
#     parser.add_argument('--ep',
#         type=int,
#         default=20,
#         help='number of epochs.')
#     parser.add_argument('--bsize',
#         type=int,
#         default=512,
#         help='batch size.')
#     parser.add_argument('--bsize-eval',
#         type=int,
#         default=500,
#         help='batch size.')
#     # GPU configs
#     parser.add_argument('--dev-place',
#         type=bool,
#         default=False,
#         help='log device placement.')
#     # Log configs
#     parser.add_argument('--model-dir',
#         type=str,
#         default='./log',
#         help='directory where model parameters, graph, etc are saved.')
#     parser.add_argument('--model-path',
#         type=str,
#         default=None,
#         help='Pre-trained model path.')
#     parser.add_argument('--data-dir',
#         type=str,
#         default='../data/cifar10')
#     args, unparsed = parser.parse_known_args()
#     if len(unparsed) != 0:
#         raise SystemExit('Unknown argument: {}'.format(unparsed))
#     main(args)

"""model of vgg-net 16"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import vgg_base

class VGG16(vgg_base.ConvNet):
    """docstring for VGG16""" 
    def __init__(self, is_training,batch_norm_decay,batch_norm_epsilon,data_format='channels_first'):
        super(VGG16,self).__init__(is_training,data_format,batch_norm_decay,batch_norm_epsilon)
        self.num_classes = 10+1 
        self.block_num=2
        self.block_filter_sizes = [256,512]

    def forward_pass(self,x,input_data_format='channel_last'):
        if self._data_format != input_data_format:
            if input_data_format == 'channel_last':
                x = tf.transpose(x,[0,3,1,2])
            else:
                x = tf.transpose(x,[0,2,3,1])
        x = x/128.0 -1 
        x = self._conv_layer(x,3,64,1)
        x = self._conv_layer(x,3,64,1)
        x = self._max_pool_layer(x,2,1)
        x = self._conv_layer(x,3,128,1)
        x = self._conv_layer(x,3,128,1)
        x = self._max_pool_layer(x,2,1)
        for bfs in self.block_filter_sizes:
            for i in range(self.block_num):
                x = self._conv_layer(x,3,bfs,1)
            x = self._max_pool_layer(x,2,2)
        x = self._full_connected_layer(x,512)
        #x = self._full_connected_layer(x,256)
        x = self._full_connected_layer(x,self.num_classes)
        x = self._softmax_layer(x)

        return x

        

        