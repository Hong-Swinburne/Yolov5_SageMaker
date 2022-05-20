import os
from os.path import exists, join
import argparse
import logging
import json
import urllib3
import ast



CONST_PRINT_TAB = "\t"
CONST_PRINT_TAB_WARN = "\t"
g_SLACK_APP_MSG = "None"
g_JOB_NAME = "None"

# @brief: send one message to slack channel by web hook.
# @in: hook url
# @in: msg
# @in: divider
# @ret: job status
# @birth: v0.0.7
def fnSendMsg2Slack(hook, msg, divider="-"*80):
    try:
        print("-------> {}".format(msg))     # <-- for cloudwatch.
        
        http = urllib3.PoolManager()
        data = { "text": divider+'\n'+str(msg)+'\n' }
        ret = http.request("POST",
                         hook,
                         body = json.dumps(data),
                         headers = {
                             "Content-Type": "application/json"
                         })
    except:
        print("[exception] Fail to send message to Slack. msg: {}".format(str(msg)))
        return None

    return ret


# @brief: load temp file.
# @param[in]: file path
# @ret: file string
# @birth: v0.0.4
# @update: v0.0.4
def fnLoad_template_file(temp_file_path):
    print(CONST_PRINT_TAB + "load configure template: {}".format(temp_file_path))
    if not exists(temp_file_path):
        logging.warning(CONST_PRINT_TAB_WARN + "Invald path: {}".format(temp_file_path))
        return ""

    with open(temp_file_path) as f:
        template = f.read()
    return template


# @brief: set the values for the template
# @param[in]: template file string stream.
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @param[in]: dev config vlaue
# @ret: new file string steam
# @birth: v0.0.4
# @update: v0.0.4
def fnGenerate_config_file(template, \
        train_img, train_batch_size, train_epochs, train_data, train_weights, \
        detect_weights, detect_imgsz, detect_conf_thres, detect_source, \
        cfg, hyp, name, optimizer, freeze):

    logging.info(CONST_PRINT_TAB+"Configure the training configuration file.")

    #
    # Add one line here for a new item.
    #
    new_config = template
    new_config = new_config.replace('{train_img}',          str(train_img))
    new_config = new_config.replace('{train_batch_size}',   str(train_batch_size))
    new_config = new_config.replace('{train_epochs}',       str(train_epochs))
    new_config = new_config.replace('{train_data}',         str(train_data))
    new_config = new_config.replace('{train_weights}',      str(train_weights))
    new_config = new_config.replace('{detect_weights}',     str(detect_weights))
    new_config = new_config.replace('{detect_imgsz}',       str(detect_imgsz))
    new_config = new_config.replace('{detect_conf_thres}',  str(detect_conf_thres))
    new_config = new_config.replace('{detect_source}',      str(detect_source))
    new_config = new_config.replace('{cfg}',                str(cfg))
    new_config = new_config.replace('{hyp}',                str(hyp))
    new_config = new_config.replace('{name}',               str(name))
    new_config = new_config.replace('{optimizer}',          str(optimizer))
    new_config = new_config.replace('{freeze}',             str(freeze))

    return new_config


# @brief: save yaml file.
# @param[in]: file string
# @param[in]: save path
# @ret: status
# @birth: v0.0.4
# @update: v0.0.4
def fnSave_config(config, fpath):
    print(CONST_PRINT_TAB + "save config file: {}".format(fpath))
    with open(fpath, 'w') as f:
        f.write(config)
    
    return True


# @brief: generate dvc yaml by template file.
# @param[in]: generated yaml path.
# @param[in]: template file path.
# @param[in]: values for yaml file.
# @ret: new yaml file path
# @birth: v0.0.4
# @update: v0.0.4
def generate_dvc_yaml(const_dvc_generated_config_dirPath, const_dvc_config_template_path, opt_dict):

    template_file = fnLoad_template_file( const_dvc_config_template_path )
    # print(template_file)

    #
    # Add one line here for a new item.
    #
    train_img          = opt_dict["train_img"]
    train_batch_size   = opt_dict["train_batch_size"]
    train_epochs       = opt_dict["train_epochs"]
    train_data         = opt_dict["train_data"]
    train_weights      = opt_dict["train_weights"]
    detect_weights     = opt_dict["detect_weights"]
    detect_imgsz       = opt_dict["detect_imgsz"]
    detect_conf_thres  = opt_dict["detect_conf_thres"]
    detect_source      = opt_dict["detect_source"]
    cfg                = opt_dict["cfg"]
    hyp                = opt_dict["hyp"]
    name               = opt_dict["name"]
    optimizer          = opt_dict["optimizer"]
    freeze             = opt_dict["freeze"]

    new_config = fnGenerate_config_file(template_file, \
        train_img, train_batch_size, train_epochs, train_data, train_weights, \
        detect_weights, detect_imgsz, detect_conf_thres, detect_source, \
        cfg, hyp, name, optimizer, freeze)

    fnSave_config(new_config, const_dvc_generated_config_dirPath)

    # todo: if fail to generate it, return None.
    return const_dvc_generated_config_dirPath

# @brief: download the chosen object photos
# @param[in]: this is the dict of hyper-parameters for dvc.
# @ret: dvc yaml path.
# @birth: v0.1.1
# @update: v0.1.1
def prepare_dvc_resources(opt_dict):
    try:
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[DVC] {}: Prepare DVC resources.".format(g_JOB_NAME))

        # -------------------------------------------
        # DVC
        # -------------------------------------------

        const_dvc_yaml_template_path  = opt_dict["params_temp_path"]
        const_dvc_generated_yaml_path = opt_dict["params_yaml_path"]

        # Generate params.yaml and return its path.
        # So, call this function in wrapper code in your docker image and you will get the generated yaml file path.
        generated_yaml_path = generate_dvc_yaml(const_dvc_generated_yaml_path, const_dvc_yaml_template_path, opt_dict)

        with open(generated_yaml_path, "r") as the_file:
            yaml_txt = the_file.read()
            const_msg = "[DVC] {}: Generate {}.".format(g_JOB_NAME, generated_yaml_path) \
                + "\n{}".format(yaml_txt)
            fnSendMsg2Slack(g_SLACK_APP_MSG, const_msg)

    except:
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[DVC] {}: Fail to prepare DVC resources.".format(g_JOB_NAME), ">"*80 )
        raise

    return generated_yaml_path

# @brief: start the DVC pipeline
# @param[in]: the path of params.yaml.
# @param[in]: add more if necessary.
# @ret: undefined
# @birth: v0.1.1
# @update: v0.1.1
def start_dvc_pipeline(cfg_file_path):
    try:
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[DVC] {}: Start DVC pipeline.".format(g_JOB_NAME))

        # -------------------------------------------
        # DVC todo.
        # -------------------------------------------


    except:
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[DVC] {}: Fail to start DVC pipeline.".format(g_JOB_NAME), ">"*80 )
        raise

    return None


# @brief: download the chosen object photos
# @param[in]: local root path
# @param[in]: this is the dict of hyper-parameters for dvc.
# @param[in]: s3 path list of train ds folder
# @param[in]: s3 path list of test  ds folder
# @param[in]: s3 path of input folder path
# @ret: local train folder path
# @ret: local test  folder path
# @ret: local input folder path
# @birth: v0.0.1
# @update: v0.0.1
def prepare_resouces(const_SM_CODE_PATH, const_SM_DATA_PATH, opt_dict):
    
    fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: (1) Prepare resources.".format(g_JOB_NAME))
    
    const_LOCAL_DATASET_TRAIN_DIR_NAME = "powercor_dataset/train"
    const_LOCAL_DATASET_VALID_DIR_NAME = "powercor_dataset/valid"
    const_LOCAL_DATASET_TEST_DIR_NAME  = "powercor_dataset/test"
    const_LOCAL_INPUT_DIR_NAME = opt_dict["weights_name"]

    local_dataset_train_dir_path = join(const_SM_DATA_PATH, const_LOCAL_DATASET_TRAIN_DIR_NAME)
    local_dataset_valid_dir_path = join(const_SM_DATA_PATH, const_LOCAL_DATASET_VALID_DIR_NAME)
    local_dataset_test_dir_path  = join(const_SM_DATA_PATH, const_LOCAL_DATASET_TEST_DIR_NAME)
    local_dataset_input_dir_path = join(const_SM_CODE_PATH, const_LOCAL_INPUT_DIR_NAME)
    
    for dir in [const_SM_CODE_PATH, local_dataset_train_dir_path, local_dataset_valid_dir_path, local_dataset_test_dir_path]:
        os.makedirs(dir, exist_ok=True)
    
    fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: Download the dataset.".format(g_JOB_NAME))
    
    os.system("aws s3 cp {} {}".format(opt_dict["weights_path"], local_dataset_input_dir_path)) #weights
    os.system("aws s3 cp {} {}".format(opt_dict["S3data"], opt_dict["data"])) # data.yaml

    for train_dir in opt_dict["train"]:
        print(train_dir)
        os.system("aws s3 cp {} {} --recursive".format(train_dir, local_dataset_train_dir_path))
    for valid_dir in opt_dict["valid"]:
        print(valid_dir)
        os.system("aws s3 cp {} {} --recursive".format(valid_dir, local_dataset_valid_dir_path))
    for test_dir in opt_dict["test"]:
        print(test_dir)
        os.system("aws s3 cp {} {} --recursive".format(test_dir, local_dataset_test_dir_path))
   
    
    
    fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: Dataset downloaded.".format(g_JOB_NAME))



# @brief: start a training job.
# @param[in]: local train ds folder path
# @param[in]: local test  ds folder path
# @param[in]: local train folder path
# @param[in]: steps
# @param[in]: epochs
# @ret: success or failure for saving.
# @birth: v0.0.1
# @update: v0.0.1
def start_training_job(const_SM_CODE_PATH, opt_dict):
    
    fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: (2) Start the training job.".format(g_JOB_NAME))
    
    training_setting = 'image size:{}, batch:{}, hyp:{}, epochs:{}, data:{}, cfg:{}, weights:{}, name:{}, optimizer:{}'.format( \
        opt_dict['img_size'], opt_dict['batch'], opt_dict['hyp'],  opt_dict['epoch'], opt_dict['data'], opt_dict['yolo_arch'], opt_dict['weights_name'], opt_dict['name'], opt_dict['optimizer'])

    print(training_setting)
    
    const_msg = "[CONTAINER] {}: training settings.".format(g_JOB_NAME) \
        + "\n{}".format(training_setting)
    fnSendMsg2Slack(g_SLACK_APP_MSG, const_msg)
    
    cmd = 'python {}/train.py --img {} --batch {} --hyp {} --epochs {} --data {} --cfg {} --weights {} --name {} --optimizer {}'.format( \
        const_SM_CODE_PATH, opt_dict['img_size'], opt_dict['batch'], opt_dict['hyp'],  opt_dict['epoch'], opt_dict['data'], opt_dict['yolo_arch'], opt_dict['weights_name'], opt_dict['name'], opt_dict['optimizer'])
    
    const_msg = "[CONTAINER] {}: training details.".format(g_JOB_NAME) + "\n"
    fnSendMsg2Slack(g_SLACK_APP_MSG, const_msg)
    
    
    f = os.popen(cmd, "r")
    print(f.read())
    f.close()
    
        
    const_OUTPUT_DIR_PATH = "{}/runs/train/{}".format(const_SM_CODE_PATH, opt_dict['name'])

    return const_OUTPUT_DIR_PATH


# @brief: save the trained result.
# @param[in]: local output folder path
# @param[in]: s3 output folder path, which is a new version of trained model.
# @ret: success or failure for saving.
# @birth: v0.0.1
# @update: v0.0.1
def save_model(local_output_dirPath, const_S3_OUTPUT_DIR_PATH):
    
    fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: (3) Save the model.".format(g_JOB_NAME))

    # Upload training results to S3    
    cmd = 'aws s3 cp {} {} --recursive'.format(local_output_dirPath, const_S3_OUTPUT_DIR_PATH)
    
    f = os.popen(cmd, "r")
    print(f.read())
    f.close()
    
    const_msg = "[CONTAINER] {}: model saved to.".format(g_JOB_NAME) \
        + "\n{}".format(const_S3_OUTPUT_DIR_PATH)
    fnSendMsg2Slack(g_SLACK_APP_MSG, const_msg)

    return True;


# @brief: Entrance point
# @birth: v0.0.1
# @update: v0.0.1
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument('--job_name', type=str)
    parser.add_argument('--slackhook', type=str)
    
    #======================================================================================================
    parser.add_argument('--train', type=ast.literal_eval)
    parser.add_argument('--valid', type=ast.literal_eval)
    parser.add_argument('--test', type=ast.literal_eval)
    parser.add_argument('--input', type=str) # pretrained weights
    parser.add_argument('--output', type=str)

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--weights_name', type=str,  default='yolov5s.pt') # File name of training weights, e.g. yolov5s.pt

    # these for dvc.
   
    parser.add_argument('--train_img', type=int, default=640) # image size for training
    parser.add_argument('--train_batch_size', type=int, default=32) # training batch size
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--train_data', type=str) # path to data.yaml file
    parser.add_argument('--train_weights', type=str) # Full path of the training weights
    parser.add_argument('--detect_weights', type=str) 
    parser.add_argument('--detect_imgsz', type=ast.literal_eval, default=str([640,640]))
    parser.add_argument('--detect_conf_thres', type=float, default=0.25)
    parser.add_argument('--detect_source', type=str)
    parser.add_argument('--cfg', type=str, default='yolov5s') # model architecture file
    parser.add_argument('--hyp', type=str, default='scratch-low') # hyper-parameter setting file for training
    parser.add_argument('--name', type=str, default='yolov5') # experiment name, related to dataset name
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD') # optimizer
    parser.add_argument('--freeze', type=ast.literal_eval, default=str([0,0]))# freeze layers
    # parser.add_argument('--freeze', type=str, default=str(0)) # freeze layers
    #======================================================================================================
    
    args = parser.parse_args()
   

    const_SM_CODE_PATH = "/opt/ml/code/yolo5-detection"
    const_SM_DATA_PATH = "/opt/ml/input/data"
    
    #=========================== For debugging ===================================
    # args.job_name = "asset-det"
    # args.slackhook = "https://hooks.slack.com/services/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"
    # args.train = ["s3://hpan-asset-verfi-testdata/small_dataset/train"]
    # args.valid = ["s3://hpan-asset-verfi-testdata/small_dataset/valid"]
    # args.test = ["s3://hpan-asset-verfi-testdata/small_dataset/test"]
    # args.weights_name = "yolov5s.pt"
    # args.input = 's3://asset-det-train-pipeline/{}/weights/pretrained/{}'.format('yolo5', args.weights_name)
    # args.output = "s3://asset-det-train-pipeline/yolo5/training-results/"
    # args.train_weights = args.weights_name
    # args.train_data = "s3://hpan-asset-verfi-testdata/small_dataset/data.yaml"
    # args.data = '{}/{}/data.yaml'.format(const_SM_CODE_PATH, 'powercor_dataset')
    # args.hyp = 'scratch-low'
    # args.cfg = 'yolov5s'
    # args.train_img = 640
    # args.train_batch_size = 32
    # args.epoch = 2
    # args.step = 200
    # args.name = 'powercor'
    # args.optimizer = 'SGD'
    # args.freeze = [0]
    # args.train_epochs = args.epoch
    # args.train_cfg = '1'
    # args.train_hyp = '1'
    # args.train_name = '1'
    # args.train_optimizer = '1'
    # args.train_freeze = '1'
    # args.detect_weights = '1'
    # args.detect_imgsz = '1'
    # args.detect_conf_thres = '1'
    # args.detect_source = '1'
    #==============================================================================
    
    print("########### pwd & ls ##############")
    cmd='pwd'
    log = os.system(cmd)
    print(log)

    cmd='ls'
    log = os.system(cmd)
    print(log)
    
    print("########### print gpu info ##############")
    cmd='nvidia-smi'
    log = os.system(cmd)
    print(log)
    
    try: 
        print("job_name:  {}, {}".format(type(args.job_name ), args.job_name ))
        print("slackhook: {}, {}".format(type(args.slackhook), args.slackhook  ))

        # Global variables
        g_SLACK_APP_MSG = args.slackhook
        g_JOB_NAME = args.job_name

        const_msg = "[CONTAINER] {}: Receive hyper-parameters.".format(g_JOB_NAME) \
        + "\ntrain:  {}, {}".format(type(args.train ), args.train )   \
        + "\nvalid:  {}, {}".format(type(args.valid ), args.valid )   \
        + "\ntest:   {}, {}".format(type(args.test  ), args.test  )   \
        + "\ninput:  {}, {}".format(type(args.input ), args.input )   \
        + "\noutput: {}, {}".format(type(args.output), args.output)   \
        + "\nepoch:  {}, {}".format(type(args.epoch ), args.epoch )   \
        + "\nstep:   {}, {}".format(type(args.step  ), args.step  )   \
        + "\n[dvc] train_img:        {}, {}".format(type(args.train_img),         args.train_img        ) \
        + "\n[dvc] train_batch_size: {}, {}".format(type(args.train_batch_size),  args.train_batch_size ) \
        + "\n[dvc] train_epochs:     {}, {}".format(type(args.train_epochs),      args.train_epochs     ) \
        + "\n[dvc] train_data:       {}, {}".format(type(args.train_data),        args.train_data       ) \
        + "\n[dvc] train_weights:    {}, {}".format(type(args.train_weights),     args.train_weights    ) \
        + "\n[dvc] detect_weights:   {}, {}".format(type(args.detect_weights),    args.detect_weights   ) \
        + "\n[dvc] detect_imgsz:     {}, {}".format(type(args.detect_imgsz),      args.detect_imgsz     ) \
        + "\n[dvc] detect_conf_thres:{}, {}".format(type(args.detect_conf_thres), args.detect_conf_thres) \
        + "\n[dvc] detect_source:    {}, {}".format(type(args.detect_source),     args.detect_source    ) \
        + "\n[dvc] cfg:              {}, {}".format(type(args.cfg),               args.cfg              ) \
        + "\n[dvc] hyp:              {}, {}".format(type(args.hyp),               args.hyp              ) \
        + "\n[dvc] name:             {}, {}".format(type(args.name),              args.name             ) \
        + "\n[dvc] optimizer:        {}, {}".format(type(args.optimizer),         args.optimizer        ) \
        + "\n[dvc] freeze:           {}, {}".format(type(args.freeze),            args.freeze           )

        fnSendMsg2Slack(g_SLACK_APP_MSG, const_msg)

        # this is designed for dvc.
        const_PARAMS_TEMP_PATH = "./params.temp"
        const_PARAMS_YAML_PATH = "./params.yaml"

        opt_dict = {}
        opt_dict["job_name"] = args.job_name
        opt_dict["slackhook"] = args.slackhook
        
        opt_dict["train"] = args.train       # training data dirPathList on S3
        opt_dict["valid"] = args.valid       # valid data dirPathList on S3
        opt_dict["test"] = args.test         # test data dirPathList on S3
        opt_dict["weights_name"] = args.weights_name # weights name
        opt_dict["weights_path"] = args.input  # weights path   
        opt_dict["output"] = args.output       
        opt_dict["S3data"] = args.train_data
        opt_dict["data"] = '{}/{}/data.yaml'.format(const_SM_CODE_PATH, 'powercor_dataset')
        opt_dict["hyp"] = '{}/data/hyps/hyp.{}.yaml'.format(const_SM_CODE_PATH, args.hyp)   
        opt_dict["yolo_arch"] = '{}/models/{}.yaml'.format(const_SM_CODE_PATH, args.cfg)     
        opt_dict["img_size"] = args.train_img
        opt_dict["batch"] = args.train_batch_size 
        opt_dict["epoch"] = args.epoch
        opt_dict["name"] = args.name 
        opt_dict["optimizer"] = args.optimizer 
        opt_dict["freeze"] = args.freeze 

        # this is designed for dvc.
        opt_dict["train_img"]         = args.train_img
        opt_dict["train_batch_size"]  = args.train_batch_size
        opt_dict["train_epochs"]      = args.train_epochs
        opt_dict["train_data"]        = args.train_data
        opt_dict["train_weights"]     = args.input
        opt_dict["detect_weights"]    = args.detect_weights
        opt_dict["detect_imgsz"]      = args.detect_imgsz
        opt_dict["detect_conf_thres"] = args.detect_conf_thres
        opt_dict["detect_source"]     = args.detect_source
        opt_dict["cfg"]               = args.cfg
        opt_dict["params_temp_path"]  = const_PARAMS_TEMP_PATH
        opt_dict["params_yaml_path"]  = const_PARAMS_YAML_PATH
        


        # ----------------------------------------------------------------
        # DVC
        # ----------------------------------------------------------------
        
        const_NEW_PARAMS_YAML = prepare_dvc_resources(opt_dict)

        # todo for dvc pipeline here.
        start_dvc_pipeline(const_NEW_PARAMS_YAML)


        const_S3_OUTPUT_DIR_PATH = opt_dict["output"]

        # [1] step one.
        prepare_resouces(const_SM_CODE_PATH, const_SM_DATA_PATH, opt_dict)

        # [2] step two.
        local_output_dirPath = start_training_job(const_SM_CODE_PATH, opt_dict)

        # [3] step three.
        ret = save_model(local_output_dirPath, const_S3_OUTPUT_DIR_PATH)
        
        print("Successfully save the trained model on {}".format(const_S3_OUTPUT_DIR_PATH))
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: Training job finished.".format(g_JOB_NAME))

    except:
        print("[exception] fail to run the training job.")
        fnSendMsg2Slack(g_SLACK_APP_MSG, "[CONTAINER] {}: Fail to run the training job.".format(g_JOB_NAME), ">"*80 )
        
