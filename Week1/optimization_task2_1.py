import wandb

from task1_2 import *
from task2_1 import BackgroundRemoval
from evaluation import *

from tqdm import tqdm

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'map'},
    'parameters': {
        'alpha' : {
            'distribution': 'uniform',
            'max': 15.0,
            'min': 1.0
        },

        'kernel_size': {
            'values': [3, 5, 7, 9, 11]
        },

        'rho' : {
            'distribution': 'uniform',
            'max': 1.0,
            'min': 0.0
        }

    }
 }

def evaluate(pathResults, bboxGT, kernel_open, kernel_close):
    bboxPred = get_all_bb(pathResults, kernel_open, kernel_close)

    map, mrecs, mprecs = mAP(bboxPred, bboxGT)

    return map, mrecs, mprecs

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        inFrames = 'c010/vdo.avi'
        outFrames = 'framesOriginal'

        kernel_number = config.kernel_size
        kernel_open = 3 #NEW
        kernel_close = 30 #NEW

        # modelo.train()
        modelo.fast_test(alpha=config.alpha, ro = config.rho, kernel_size=(kernel_number, kernel_number))

        n_frames = len(os.listdir(outFrames))
        xml_file = 'ai_challenge_s03_c010-full_annotation.xml'
        classes = ['car'] # The other class is bike
        bbox_info = read_ground_truth(xml_file, classes, n_frames)

        outputFolderModel = 'framesResult_adaptive/'
        
        mapScore, precScore, recScore = evaluate(outputFolderModel, bbox_info, kernel_open, kernel_close)

        wandb.log({'map': mapScore, 'precision': precScore, 'recall': recScore})


if __name__ == '__main__':

    inFrames = 'c010/vdo.avi'
    outFrames = 'framesOriginal'

    kernel_number = (11, 11)
    modelo = BackgroundRemoval(inFrames, outFrames, alpha = 0.25, ro = 0.5, morph = True, kernel_size=(kernel_number, kernel_number))

    modelo.train()

    sweep_id = wandb.sweep(sweep_config, project="OptimizationC6_Task2_1")
    wandb.agent(sweep_id, function=train, count=200)
