import wandb

from task1_2 import *
from task2_1 import BackgroundRemoval
from evaluation import *

from tqdm import tqdm
import math

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'map'},
    'parameters': {
        'alpha' : {
            'distribution': 'uniform',
            'max': 15.0,
            'min': 1.0
        },

        'kernel_open': {
            'values': [3, 5, 7, 9, 11]
        },

        'kernel_close': {
            'values': [3, 5, 7, 9, 11]
        },

        'rho' : {
            'distribution': 'uniform',
            'max': 0.2,
            'min': 0.0
        }

    }
 }

def evaluate(pathResults, bboxGT, kernel_open, kernel_close):
    bboxPred = get_all_bb(pathResults,  kernel_open, kernel_close)

    numFrame = math.floor(len(os.listdir(pathResults)) * 0.25)

    map, mrecs, mprecs = mAP(bboxPred[numFrame:], bboxGT[numFrame:])

    return map, mrecs, mprecs

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        inFrames = 'c010/vdo.avi'
        outFrames = 'framesOriginal'

        kernel_open = config.kernel_open #New
        kernel_close = config.kernel_close #New

        modelo = BackgroundRemoval(inFrames, outFrames, alpha = 0.25, ro = 0.5, morph = False, kernel_size=(kernel_number, kernel_number))
        modelo.train()
        modelo.fast_test(alpha=config.alpha, ro = config.rho)

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

    sweep_id = wandb.sweep(sweep_config, project="OptimizationC6_Task2_1v4")
    wandb.agent(sweep_id, function=train, count=200)
