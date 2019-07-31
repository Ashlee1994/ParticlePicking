import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os,sys,shutil,time
from vgg19 import Vgg19
from utils import load_train
from args import Train_Args

def train():
    args = Train_Args() ## load args parameters to a class named args
    train_start = time.time()
    time_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    checkpoint_dir = args.model_save_path
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    output_name = args.model_save_path + "training_messages.txt"
    output = open(output_name, 'w')
    print("read data start.",flush=True)
    output.write("read data start.\n")

    train_x, train_y, test_x, test_y = load_train(args)
    print("shape of train_x: " , train_x.shape)
    time_end = time.time()
    print("\nread done! totally cost: %.5f \n" %(time_end - time_start),flush=True)
    output.write("read done! totally cost: " + str(time_end - time_start) +'\n')
    output.flush()
    print("training start.",flush=True)
    # copy argument file
    shutil.copyfile(args.args_filename,args.model_save_path + args.args_filename)
    shutil.copyfile(args.model_filename,args.model_save_path + args.model_filename)
    shutil.copyfile(args.train_filename,args.model_save_path + args.train_filename)

    time_start = time.time()
    tot_cost = []
    plot = []
    plot_train = []
    best_test_accuracy = 0
    best_train_accuracy = 0

    # training start
    with tf.Session() as sess:
        train_num_batch_gpus = len(train_x)
        print("num_batch is %d" % train_num_batch_gpus,flush=True)
        test_num_batch_gpus = len(test_x)
        decay_step = train_num_batch_gpus * args.lr_decay_epoch

        model = Vgg19(args,decay_step)
        sess.run(tf.global_variables_initializer())
        print("train size is %d " % len(train_x), flush=True)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10000)
        if args.checkpoint is not None:
            print('Load checkpoint %s' % args.checkpoint)
            saver = tf.train.Saver()
            # saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Restore model failed!',flush=True)
            # saver.restore(sess, args.checkpoint)
            # init_step = global_step.eval(session=sess)

        for step in range(args.num_epochs):
            print('\n=============== Epoch %d/%d ==============='% (step + 1,args.num_epochs),flush=True)
            output.write("\n=============== Epoch " + str(step + 1) + "/" + str(args.num_epochs) + " ===============\n")
            start_time = time.time()
            # Train
            train_loss = 0
            train_acc = 0
            for i in range(train_num_batch_gpus):
                batch_x = train_x[i]
                batch_y = train_y[i]
                loss_value,acc_value,lr,_= sess.run([model.loss, model.acc,model.lr, model.train_op], {model.X:batch_x, model.Y: batch_y})
                # print('(Training): acc_value = %.4f' % acc_value , flush=True)
                train_loss += loss_value
                train_acc += acc_value

            train_loss /= train_num_batch_gpus
            tot_cost.append(train_loss)
            train_acc /= train_num_batch_gpus
            duration = time.time() - start_time

            if step % args.display_interval == 0 or step < 10:
                num_examples_per_step = args.batch_size * args.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.6f, acc=%.6f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, train_loss, train_acc, lr,
                                     examples_per_sec, sec_per_batch),flush=True)
                format_str = ('%s: (Training) step %d, loss=%.6f, acc=%.6f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)\n')
                output.write(format_str % (datetime.now(), step, train_loss, train_acc, lr,
                                     examples_per_sec, sec_per_batch))
                output.flush
                # summary_writer.add_summary(train_summary_str, step)


            if train_acc > best_train_accuracy:
                best_train_accuracy = train_acc

            plot_train.append(1- train_acc)

            # val
            if step % args.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(test_num_batch_gpus):
                    batch_x = test_x[i]
                    batch_y = test_y[i]
                    loss_value, acc_value = sess.run([model.loss, model.acc],
                                feed_dict={model.X:batch_x,model.Y:batch_y})
                    val_loss += loss_value
                    val_acc += acc_value

                best_test_accuracy_tmp = max(best_test_accuracy, val_acc)
                val_loss /= test_num_batch_gpus
                val_acc /= test_num_batch_gpus
                plot.append(1- val_acc)
                format_str = ('%s: (val)      step %d, loss=%.6f, acc=%.6f')
                print (format_str % (datetime.now(), step, val_loss, val_acc),flush=True)
                format_str = ('%s: (val)      step %d, loss=%.6f, acc=%.6f\n')
                output.write (format_str % (datetime.now(), step, val_loss, val_acc))
                output.flush

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=best_test_accuracy_tmp)
                # summary_writer.add_summary(val_summary, step)
                # summary_writer.flush()

            # Save the model checkpoint periodically.
            # if (step > init_step and step % args.checkpoint_interval == 0) or (step + 1) == args.max_steps:
            if (val_acc > best_test_accuracy ) or (step + 1) == args.num_epochs:
                best_test_accuracy = max(best_test_accuracy, val_acc)
                checkpoint_path = os.path.join(args.model_save_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            print("best_train_accuracy: %.6f,  best_test_accuracy: %.6f" % ( best_train_accuracy,best_test_accuracy) ,flush=True)
            output.write("best train accuracy: " + str(best_train_accuracy) + '\n')
            output.write("best test accuracy: " + str(best_test_accuracy) + '\n')
            output.flush()

        time_end = time.time()
        print("\ntraining done! totally cost: %.5f \n" %(time_end - time_start),flush=True)
        output.write("training done! totally cost: " + str(time_end - time_start) + '\n')
        output.flush()

    train_end = time.time()
    print("\ntrain done! totally cost: %.5f \n" %(train_end - train_start),flush=True)
    print("best_train_accuracy: %.6f " % best_train_accuracy)
    print("best_test_accuracy: %.6f " % best_test_accuracy)
    output.write("best_train_accuracy: " + str(best_train_accuracy) + '\n')
    output.write("best_test_accuracy: " + str(best_test_accuracy) + '\n')
    output.flush()
    output.close

    # draw picture of loss and error
    plt.subplot(211)
    plt.title('error red:test  green:train')
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid(True)
    plt.plot(range(len(plot)),plot,"r^--",range(len(plot_train)),plot_train,"g^--")
    
    plt.subplot(212)
    plt.title('loss of training')
    plt.plot(range(len(tot_cost)),tot_cost,"b^--")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(args.model_save_path + "loss_error.png")
    plt.show()

if __name__ == '__main__':
    train()

