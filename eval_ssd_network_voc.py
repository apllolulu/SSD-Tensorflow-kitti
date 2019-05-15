
"""Generic evaluation script that evaluates a SSD model
on a given dataset."""
import math
import sys
import six
import time

import numpy as np
import tensorflow as tf
import tf_extended as tfe
import tf_utils
from tensorflow.python.framework import ops

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

"""
https://github.com/balancap/SSD-Tensorflow

"""
############## 2007 test model 2007
"""
DATASET_DIR=./data_tfrecords/VOC2007/test
TRAIN_DIR=./logs/logs_VOC2007/
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --max_num_batches=250
    
# max_num_batches = 100 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:04:09
AP_VOC07/mAP[0.76568520419043606]
AP_VOC12/mAP[0.77051671874346617]

# max_num_batches = 150 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:57:19
AP_VOC07/mAP[0.78881872920367646]
AP_VOC12/mAP[0.80046833524442573]

# max_num_batches = 200 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:55:31
AP_VOC07/mAP[0.786304143696222]
AP_VOC12/mAP[0.80166423277882792]

# max_num_batches = 250 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-09:01:02
AP_VOC07/mAP[0.79016630675639588]
AP_VOC12/mAP[0.80714835148506558]


# max_num_batches = 300 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:02:36
AP_VOC07/mAP[0.77983190480346476]
AP_VOC12/mAP[0.79904817462099043]

# max_num_batches = 300 batch_size=2 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:43:11
AP_VOC07/mAP[0.74407434380631]
AP_VOC12/mAP[0.7638786682418427]

# max_num_batches = 300 batch_size=4 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:06:20
AP_VOC07/mAP[0.71771184805098165]
AP_VOC12/mAP[0.74011616546018733]

# max_num_batches = 300 batch_size=8 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:09:01
AP_VOC07/mAP[0.72665661630532719]
AP_VOC12/mAP[0.74648703133708372]

# max_num_batches = 400 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:48:00
AP_VOC07/mAP[0.77734552711203719]
AP_VOC12/mAP[0.795227031737687]


# max_num_batches = 500 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-09-13:52:27
AP_VOC07/mAP[0.758332184908063]
AP_VOC12/mAP[0.77779746572523845]

# max_num_batches = 500 batch_size=8 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-08:00:57
AP_VOC07/mAP[0.72664179282122432]
AP_VOC12/mAP[0.74680320852475757]

# max_num_batches = 800 batch_size=1 model.ckpt-0
INFO:tensorflow:Finished evaluation at 2018-09-10-07:53:53
AP_VOC07/mAP[0.72004612599501883]
AP_VOC12/mAP[0.74360586486466262]



"""
############## 2007 test model 2012
"""
DATASET_DIR=./data_tfrecords/VOC2007/test
TRAIN_DIR=./logs/logs_voc2012/
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network_voc.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=False \
    --batch_size=8 \
    --max_num_batches=250



# max_num_batches = 250 batch_size=1 model.ckpt-117170
INFO:tensorflow:Finished evaluation at 2018-09-10-10:56:53
AP_VOC07/mAP[0.77849502209865051]
AP_VOC12/mAP[0.79261317387971619]


# model.ckpt-189207
INFO:tensorflow:Finished evaluation at 2018-09-10-11:53:35
AP_VOC07/mAP[0.778873116693061]
AP_VOC12/mAP[0.79232619763676815]

# ssd_300_vgg.ckpt max_num_batches=250
INFO:tensorflow:Finished evaluation at 2018-09-10-13:31:34
AP_VOC07/mAP[0.87182771129550918]
AP_VOC12/mAP[0.8886719050965054]

# ssd_300_vgg.ckpt max_num_batches=500
INFO:tensorflow:Finished evaluation at 2018-09-10-13:35:40
AP_VOC07/mAP[0.84872660445545689]
AP_VOC12/mAP[0.87284587350100906]

# VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt max_num_batches=250
INFO:tensorflow:Finished evaluation at 2018-09-10-13:43:33
AP_VOC07/mAP[0.79738032594619446]
AP_VOC12/mAP[0.81291022012839909]

# VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt max_num_batches=500
INFO:tensorflow:Finished evaluation at 2018-09-10-13:45:11
AP_VOC07/mAP[0.76396786411238171]
AP_VOC12/mAP[0.78339897801506586]

# VGG_VOC0712_SSD_300x300_iter_120000.ckpt max_num_batches=250
INFO:tensorflow:Finished evaluation at 2018-09-10-13:47:43
AP_VOC07/mAP[0.60252309848046881]
AP_VOC12/mAP[0.61079886588173871]

# VGG_VOC0712_SSD_300x300_iter_120000.ckpt max_num_batches=500
INFO:tensorflow:Finished evaluation at 2018-09-10-13:49:00
AP_VOC07/mAP[0.58003532614540609]
AP_VOC12/mAP[0.58969474756553941]

# VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt max_num_batches=250
INFO:tensorflow:Finished evaluation at 2018-09-10-13:54:43
AP_VOC07/mAP[0.82742817897646281]
AP_VOC12/mAP[0.84558121885332371]

# VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt max_num_batches=500
INFO:tensorflow:Finished evaluation at 2018-09-10-13:56:32
AP_VOC07/mAP[0.79072826244807592]
AP_VOC12/mAP[0.81590897219294267]

####################### KITTI 测试

DATASET_DIR=./kitti_test_tfrecords
TRAIN_DIR=./logs/kitti_log/
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network_voc.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=kitti \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=False \
    --batch_size=8 \
    --max_num_batches=250

########################################

DATASET_DIR=./data_tfrecords/VOC2007/test
TRAIN_DIR=./logs/logs_voc2012/
EVAL_DIR=${TRAIN_DIR}/eval
python eval_ssd_network_voc.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${TRAIN_DIR} \
    --wait_for_checkpoints=False \
    --batch_size=8 \
    --max_num_batches=250





"""

slim = tf.contrib.slim

# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
    'remove_difficult', True, 'Remove difficult objects from evaluation.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_boolean(
    'wait_for_checkpoints', False, 'Wait for new checkpoints in the eval loop.')


FLAGS = tf.app.flags.FLAGS

def flatten(x):
    result = []
    for el in x:
        if isinstance(el,tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        # =================================================================== #
        # Dataset + SSD model + Pre-processing
        # =================================================================== #
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Get the SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)

        # Evaluation shape and associated anchors: eval_image_size
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        tf_utils.print_configuration(FLAGS.__flags, ssd_params,
                                     dataset.data_sources, FLAGS.eval_dir)
        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    common_queue_capacity=2 * FLAGS.batch_size,
                    common_queue_min=FLAGS.batch_size,
                    shuffle=False)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
            if FLAGS.remove_difficult:
                [gdifficults] = provider.get(['object/difficult'])
            else:
                gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)

            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes, gbbox_img = \
                image_preprocessing_fn(image, glabels, gbboxes,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)

            # Encode groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = \
                ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] * 5 + [len(ssd_anchors)] * 3

            # Evaluation batch.
            r = tf.train.batch(
                tf_utils.reshape_list([image, glabels, gbboxes, gdifficults, gbbox_img,
                                       gclasses, glocalisations, gscores]),
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size,
                dynamic_pad=True)
            (b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses,
             b_glocalisations, b_gscores) = tf_utils.reshape_list(r, batch_shape)

        # =================================================================== #
        # SSD Network + Ouputs decoding.
        # =================================================================== #
        dict_metrics = {}
        arg_scope = ssd_net.arg_scope(data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = \
                ssd_net.net(b_image, is_training=False)
        # Add losses functions.
        ssd_net.losses(logits, localisations,
                       b_gclasses, b_glocalisations, b_gscores)

        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
            # Detected objects from SSD output.
            localisations = ssd_net.bboxes_decode(localisations, ssd_anchors)
            rscores, rbboxes = \
                ssd_net.detected_bboxes(predictions, localisations,
                                        select_threshold=FLAGS.select_threshold,
                                        nms_threshold=FLAGS.nms_threshold,
                                        clipping_bbox=None,
                                        top_k=FLAGS.select_top_k,
                                        keep_top_k=FLAGS.keep_top_k)
            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          b_glabels, b_gbboxes, b_gdifficults,
                                          matching_threshold=FLAGS.matching_threshold)

        # Variables to restore: moving avg. or normal weights.
        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        # =================================================================== #
        # Evaluation metrics.
        # =================================================================== #
        with tf.device('/device:CPU:0'):
            dict_metrics = {}
            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                # op = tf.Print(op, [metric[0]], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                tp_fp_metric[1][c])

            # Add to summaries precision/recall values.
            aps_voc07 = {}
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

                # Average precision VOC07.
                v = tfe.average_precision_voc07(prec, rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc07[c] = v

                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v

            # Mean average precision AP_KITTI01.
            summary_name = 'AP_KITTI01/mAP'
            mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

            # Mean average precision AP_KITTI02.
            summary_name = 'AP_KITTI02/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # for i, v in enumerate(l_precisions):
        #     summary_name = 'eval/precision_at_recall_%.2f' % LIST_RECALLS[i]
        #     op = tf.summary.scalar(summary_name, v, collections=[])
        #     op = tf.Print(op, [v], summary_name)
        #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        # =================================================================== #
        # Evaluation loop.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Number of batches...
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if not FLAGS.wait_for_checkpoints:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_path)

            # Standard evaluation loop.
            start = time.time()
            slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                variables_to_restore=variables_to_restore,
                session_config=config)
            # Log time spent.
            elapsed = time.time()
            elapsed = elapsed - start
            print('******************************')
            print('Time spent : %.3f seconds.' % elapsed)
            print('******************************')
            
            print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
            print('Time spent per PICTURE: %.3f seconds.' % (elapsed / num_batches/(FLAGS.batch_size)))
            per_pic_time  = elapsed / num_batches/(FLAGS.batch_size)
            pro_fps = int(1/per_pic_time)
            print('******************************')
            print('The fps is: %d .' % (pro_fps))
            
        else:
            checkpoint_path = FLAGS.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_path)

            # Waiting loop.
            slim.evaluation.evaluation_loop(
                master=FLAGS.master,
                checkpoint_dir=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=flatten(list(names_to_updates.values())),
                variables_to_restore=variables_to_restore,
                eval_interval_secs=60,
                max_number_of_evaluations=np.inf,
                session_config=config,
                timeout=None)


if __name__ == '__main__':
    tf.app.run()
