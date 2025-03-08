import os.path as osp
JOINT = {
    'aloha':21,
    'glove':12 
}

STEP = {
    'aloha':10,    
    'glove':10
}

EPOCH = {
    'aloha':50,
    'glove':50
}

CUBE = {
    'aloha':[250, 250, 250],
    'glove':[250, 250, 250],
}


class Config(object):
    phase = 'train'
    root_dir = 'data_zhuazi' # set dataset root path

    train_file = None

    crop = False

    mode = 'RGB'

    net = 'KPFusion-resnet-18' #['KPFusion-resnet-18', 'KPFusion-convnext-T']

    dataset = 'aloha'  # ['aloha', 'glove']
    ho3d_version = 'v2'
    model_save = ''
    save_dir = './'
    # dexycb_setup = 'data_nolight'
    pretrain = '1k'
    point_num = 1024

    # load_model = '/home/yan/KP_CODE/checkpoint/data_vit_new/24/home/yan/KP_CODE/data_vit_new_sample/16k.json/KPFusion-resnet-18_ips128/best.pth'
    load_model = ''
    finetune_dir = ''

    gpu_id = '0'

    joint_num = JOINT[dataset]

    batch_size = 64
    input_size = 128
    cube_size = CUBE[dataset]
    center_type = 'refine'
    loss_type = 'L1Loss'  # ['L1Loss', 'Mse','GHM']
    augment_para = [10, 0.2, 180]
    color_factor = 0.4

    lr = 8e-4
    max_epoch = EPOCH[dataset]
    step_size = STEP[dataset]
    opt = 'adamw'  # ['sgd', 'adam']
    scheduler = 'step'  # ['auto', 'step', 'constant']
    downsample = 2 # [1,2,4,8]

    seed = 0

    awr = True
    coord_weight = 100
    deconv_weight = 1
    spatial_weight = [10,10,10]
    spatial_epoch = [24, 24, 24]

    # for AWR backbone
    feature_type = ['weight_offset']  #['weight_offset', 'weight_pos','heatmap_depthoffset','plainoffset_depth','plainoffset_depthoffset', 'offset']
    feature_para = [0.8]

    stage_type = [1]  # Depth backbone, RGB backbone, (RGB KFAM, Depth KFAM,) (RGB KFAM, Depth KFAM)

    mano_path = osp.join('./util', 'manopth')


opt = Config()
