class Train_Args():
    is_training                 =       True
    boxsize                     =       120
    dim_x                       =       1919 # micrograph width
    dim_y                       =       1855
    name_length                 =       5
    name_prefix                 =       "image_"
    name_postfix                =       ""
    mic_path                    =       "./settrain/mrc_file/"
    model_save_path             =       "./model/model_2_gpu1/"  # end with /
    positive1_box_path          =       "./settrain/positive/"
    negative1_box_path          =       "./settrain/negative/"

    args_filename               =       "args.py"  # filename of this document
    model_filename              =       "vgg19.py"
    train_filename              =       "train.py"

    gpu_device                  =       "0" # gpu id 
    num_gpus                    =       1 # number of gpus for training

    checkpoint                  =       ""  # path of previous or half-trained model
    checkpoint_interval         =       1  # number of epoches to save your model
    val_interval                =       1  # number of epoches to run test data
    display_interval            =       1  # number of epoches to display train messages

    positive_mic_start_num      =       0  
    positive_mic_end_num        =       36
    negative_mic_start_num      =       0
    negative_mic_end_num        =       36
    num_positive                =       893  # number of positive particles
    num_negative                =       926 

    rotation_angel              =       90
    rotation_n                  =       4
    num_p_test                  =       200
    num_n_test                  =       200

    regularization              =       True
    reg_rate                    =       0.001
    dropout                     =       True
    dropout_rate                =       0.5
    optimizer                   =       "adam" #  "sgd" or "mom"
    momentum                    =       0.9    # if optimizer = "mom" , set a momentum
    finetune                    =       False

    learning_rate               =       0.00001   # important!!

    batch_size                  =       48 
    num_epochs                  =       200

    decay_rate                  =       0.96
    lr_decay_epoch              =       5

class Predict_Args():
    is_training                 =       False
    name_length                 =       5
    name_prefix                 =       "image_"
    name_postfix                =       ""
    data_path                   =       "./settrain/mrc_file/"
    result_path                 =       "./result/model_5_gpu3_7/"
    model_save_path             =       "./model_wl/model_5_gpu4/"
    boxsize                     =       120     
    start_mic_num               =       0 
    end_mic_num                 =       36
    dim_x                       =       1919
    dim_y                       =       1855
    gpu_device                  =       "0,1"
    num_gpus                    =       2
    scan_step                   =       10
    accuracy                    =       0.99   ####
    threhold                    =       0.1   ####
    batch_size                  =       48
