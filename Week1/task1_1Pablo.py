from pyexpat import model
from webbrowser import get
import cv2
import os
import shutil
import math
import numpy as np
from tqdm import tqdm
import pickle

COLOR_CHANGES = {
    'grayscale' : cv2.COLOR_BGR2GRAY,
    'hsv' : cv2.COLOR_BGR2HSV
}
def getFrames(pathInput, pathOutput):
        # Uncomment this if you want to introduce another different video from the test
        if not os.path.exists(pathOutput):
            print('Creating the fodler to store the original frames.')
            cap = cv2.VideoCapture(pathInput)

            if os.path.exists(pathOutput):  
                shutil.rmtree(pathOutput)

            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


            os.mkdir(pathOutput)

            totalFrames = 0
            for _ in tqdm(range(0, frames)):
                ret, frame = cap.read()
                if ret == False:
                    break
                cv2.imwrite(pathOutput + '/frame' + str(totalFrames).zfill(5) + '.png', frame)
                totalFrames += 1

            return totalFrames, os.listdir(pathOutput)
        
        else:
            print('The folder already exists, reading the content.')
            return len([name for name in os.listdir(pathOutput) if os.path.isfile(os.path.join(pathOutput, name))]), os.listdir(pathOutput)

class BackgroundRemoval:

    def __init__(self, pathInput, pathOutput):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.numFrames, self.listFrames = getFrames(self.framesInPath, self.framesOutPath)

    def getNumFrames(self):
        return self.numFrames
    
    def getFrames(self):
        return self.listFrames
    
    def train(self, percentageOfFrames=0.25):
        lastFrame = math.floor(self.numFrames * percentageOfFrames)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = []
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_GRAYSCALE)
            frames.append(frame)

        frames = np.array(frames)

        self.meanImage = np.mean(frames, axis=0) / 255
        self.stadImage = np.std(frames, axis=0) / 255

        # cv2.imshow('Mean Image', self.meanImage)
        # cv2.waitKey(0)

    def test(self, alpha = 2.5):

        outputFolder = 'framesResult'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_GRAYSCALE)
            frame = frame * (1. / 255)

            frame = np.abs(frame - self.meanImage)

            result = frame >=  alpha * (alpha * self.stadImage + 2/255)
            result = result.astype(np.uint8) * 255
            
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(outputFolder + '/frame' + str(i).zfill(5) + '.png', result)


            
            

if __name__ == '__main__':
    # Read a video and save the frames on a folder
    inFrames = 'c010\\vdo.avi'
    outFrames = 'framesOriginal'
    
    modelo = BackgroundRemoval(inFrames, outFrames)

    modelo.train()
    modelo.test()



    