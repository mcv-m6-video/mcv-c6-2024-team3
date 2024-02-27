import wandb

from task1_2 import *
from task1_1 import BackgroundRemoval
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
        }

    }
 }

def evaluate(pathResults, bboxGT):
    bboxPred = get_all_bb(pathResults)

    map, mrecs, mprecs = mAP(bboxPred, bboxGT)

    return map, mrecs, mprecs

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        inFrames = 'c010/vdo.avi'
        outFrames = 'framesOriginal'

        kernel_number = config.kernel_size
        # modelo = BackgroundRemoval(inFrames, outFrames, alpha = config.alpha, morph = True, kernel_size=(kernel_number, kernel_number))

        # modelo.train()
        modelo.fast_test(alpha=config.alpha, kernel_size=(kernel_number, kernel_number))

        n_frames = len(os.listdir(outFrames))
        xml_file = 'ai_challenge_s03_c010-full_annotation.xml'
        classes = ['car'] # The other class is bike
        bbox_info = read_ground_truth(xml_file, classes, n_frames)

        outputFolderModel = 'framesResult/'
        
        mapScore, recScore, precScore = evaluate(outputFolderModel, bbox_info)

        wandb.log({'map': mapScore, 'precision': precScore, 'recall': recScore})


if __name__ == '__main__':
    
    inFrames = 'c010/vdo.avi'
    outFrames = 'framesOriginal'

    kernel_number = (11, 11)
    modelo = BackgroundRemoval(inFrames, outFrames, alpha = 0.25, morph = True, kernel_size=(kernel_number, kernel_number))
    modelo.train()

    sweep_id = wandb.sweep(sweep_config, project="OptimizationC6_Task1_1")
    wandb.agent(sweep_id, function=train, count=200)