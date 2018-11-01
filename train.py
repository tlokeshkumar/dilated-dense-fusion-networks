# Self written Functions
from input_utils import segmentation_data
from model import build_model, loss_funcs

# Libraries
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K_B
from os.path import exists
import coloredlogs
from glob import glob
from natsort import natsorted 
import tensorflow 

parser = argparse.ArgumentParser(description="Inputs to the code")

parser.add_argument("--dataset",type=str,help="path to master data folder")
parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='./checkpoints/model_ckpt',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='./checkpoints',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
parser.add_argument("--no_iter",type=int,default=10,help="number of epochs for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")

def get_input_lists(path):
    '''
    Returns the list of paths of images and datasets
    '''
    gt = natsorted(glob(path+"/**/GT_*.PNG"))
    ns = natsorted(glob(path+'/**/NOISY_*.PNG'))
    return gt, ns

args = parser.parse_args()
img_rows = 256
img_cols = 256

if __name__ == '__main__':
    runopts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    coloredlogs.install(level='DEBUG')
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        init = tf.global_variables_initializer()

        gt, ns = get_input_lists(args.dataset)
        
        next_element, init_op = segmentation_data(ns, gt,img_rows, img_cols, augment=True, shuffle_data=True, batch_size=args.batch_size)

        x = tensorflow.keras.Input(shape=(img_rows, img_cols, 3))
        # x = tf.placeholder(tf.float32, shape=[None, img_rows, img_cols, 3])        
        noise, ground_truth = next_element # Splitting the next element 
        m = build_model(x)
        # m = build_model(x)
        # m.input = noise

        # print (m.summary())
        
        loss = loss_funcs(m, ground_truth)

        tf.summary.image('Input-Image', noise)
        tf.summary.image('Pred', noise)
        tf.summary.image('Ground-Truth', ground_truth)

        global_step_tensor = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_step = optimizer.minimize(loss)
        with K_B.get_session() as sess:

            sess.run(init)
            sess.run(init_op)
            
            summary_writer = tf.summary.FileWriter(args.log_directory, sess.graph)
            summary = tf.summary.merge_all()

            saver = tf.train.Saver()

            tf.logging.info("TensorBoard events written to " + args.log_directory)

            if args.load_ckpt is not None:
                if exists(args.load_ckpt):
                    if tf.train.latest_checkpoint(args.load_ckpt) is not None:
                        tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(args.load_ckpt))
                        saver.restore(sess, tf.train.latest_checkpoint(args.load_ckpt))

                    else:
                        tf.logging.info('Training from Scratch -  No Checkpoint found')
        
            else:
                tf.logging.info("Training from Scratch")

            tf.logging.info('Training with Batch Size %d for %d epochs'%(args.batch_size,args.no_iter))

            while True:    
                # Training Iterations begin
                noise_values = sess.run(noise)
                print (noise_values.shape)
                print (noise_values.dtype)
                output_ = sess.run(m.output, feed_dict={x:noise_values})
                np.save("output.npy", output_)
                np.save("noise.npy", noise_values)
                exit(0)
                global_step,_ = sess.run([global_step_tensor,train_step],options = runopts, feed_dict={x:noise_values})
                # global_step,_ = sess.run([global_step_tensor,train_step],options = runopts)
                
                if global_step%(args.display_step)==0:
                    loss_val = sess.run([loss],options = runopts)
                    tf.logging.info('Iteration: ' + str(global_step) + ' Loss: ' +str(loss_val))
                
                if global_step%(args.summary_freq)==0:
                    tf.logging.info('Summary Written')
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, global_step)
                
                if global_step%(args.save_freq)==0:
                    saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())

                if global_step > args.no_iter:
                    break