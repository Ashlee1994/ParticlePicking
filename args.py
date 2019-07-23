class Train_Args():
    is_training                 =       True
    boxsize                     =       84
    dim_x                       =       1919 # micrograph width
    dim_y                       =       1855
    name_length                 =       4
    name_prefix                 =       "3er_gfm2_gold_190517_notilt_"
    name_postfix                =       "_bin4"
    mic_path                    =       "./data/mic/"
    model_save_path             =       "./model/model_3/"  # end with /
    positive1_box_path          =       "./data/positive/"
    negative1_box_path          =       "./data/negative/"

    args_filename               =       "args.py"  # filename of this document
    model_filename              =       "vgg19.py"
    train_filename              =       "train.py"

    gpu_device                  =       "0,1,2,3" # gpu id 
    num_gpus                    =       4 # number of gpus for training

    checkpoint                  =       "./model/model_2/"  # path of previous or half-trained model
    checkpoint_interval         =       1  # number of epoches to save your model
    val_interval                =       1  # number of epoches to run test data
    display_interval            =       1  # number of epoches to display train messages

    positive_mic_start_num      =       11  
    positive_mic_end_num        =       419
    negative_mic_start_num      =       11
    negative_mic_end_num        =       419
    num_positive                =       656  # number of positive particles
    num_negative                =       659 

    rotation_angel              =       90
    rotation_n                  =       4
    num_p_test                  =       150
    num_n_test                  =       150

    regularization              =       True
    reg_rate                    =       0.001
    dropout                     =       True
    dropout_rate                =       0.5
    optimizer                   =       "adam" #  "sgd" or "mom"
    momentum                    =       0.9    # if optimizer = "mom" , set a momentum
    finetune                    =       False

    learning_rate               =       0.00001   # important!!

    batch_size                  =       100 
    num_epochs                  =       10

    decay_rate                  =       0.96
    decay_step                  =       100
    lr_decay_epoch              =       5

class Predict_Args():
    is_training                 =       False
    name_length                 =       4
    name_prefix                 =       "3er_gfm2_gold_190517_notilt_"
    name_postfix                =       "_bin4"
    data_path                   =       "./data/mic/"
    result_path                 =       "./result/"
    model_save_path             =       "./model/model_5/"
    boxsize                     =       84     
    start_mic_num               =       11 
    end_mic_num                 =       419
    dim_x                       =       1919
    dim_y                       =       1855
    gpu_device                  =       "0,1,2,3"
    num_gpus                    =       4
    scan_step                   =       10
    accuracy                    =       0.7   ####
    threhold                    =       0.7   ####
    batch_size                  =       32
