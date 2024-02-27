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
            'max': 40.0,
            'min': 0.0
        },
    }
 }

def evaluate(pathResults, bboxGT):
    bboxPred = get_all_bb(pathResults)

    map = mAP(bboxPred, bboxGT)

    return map

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        inFrames = 'c010/vdo.avi'
        outFrames = 'framesOriginal'

        modelo = BackgroundRemoval(inFrames, outFrames, alpha = config.alpha, morph = True, kernel_size=(3,3))

        modelo.train()
        modelo.test()

        n_frames = len(os.listdir(outFrames))
        xml_file = 'ai_challenge_s03_c010-full_annotation.xml'
        classes = ['car'] # The other class is bike
        bbox_info = read_ground_truth(xml_file, classes, n_frames)

        outputFolderModel = 'framesResult/'
        
        mapScore = evaluate(outputFolderModel, bbox_info)

        wandb.log({'map': mapScore})


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="OptimizationC6_Task1_1")
    wandb.agent(sweep_id, function=train, count=200)