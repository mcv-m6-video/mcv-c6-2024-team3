from pyexpat import model
from webbrowser import get
import cv2
import os
import shutil
import math
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation import *
from task1_2 import *

COLOR_CHANGES = {
    'hsv' : cv2.COLOR_BGR2HSV,
    'bgr' : None
}

def evaluate(pathResults, bboxGT, kernel_open, kernel_close):
    bboxPred = get_all_bb(pathResults,  kernel_open, kernel_close)

    numFrame = math.floor(len(os.listdir(pathResults)) * 0.25)

    map, mrecs, mprecs = mAP(bboxPred[numFrame:], bboxGT[numFrame:])

    return map, mrecs, mprecs
       

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
            file_list = sorted(os.listdir(pathOutput))
            return totalFrames, file_list
        
        else:
            print('The folder already exists, reading the content.')
            file_list = sorted([name for name in os.listdir(pathOutput) if os.path.isfile(os.path.join(pathOutput, name))])
            return len(file_list), file_list

class BackgroundRemoval:

    def __init__(self, pathInput, pathOutput, colourSpace = "bgr", resize = None, alpha = 2.5, ro = 0.01, morph = False, kernel_size = (3,3)):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.colourSpace = colourSpace
        self.resize = resize
        self.alpha = alpha
        self.ro = ro
        self.morph = morph
        self.kernel_size = kernel_size
        self.numFrames, self.listFrames = getFrames(self.framesInPath, self.framesOutPath)
        

    def getNumFrames(self):
        return self.numFrames
    
    def getFrames(self):
        return self.listFrames
    
    def train(self, percentageOfFrames=0.25):
        # REMAINS THE SAME AS IN TASK1.1
        lastFrame = math.floor(self.numFrames * percentageOfFrames)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = []
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_COLOR)

            if COLOR_CHANGES[self.colourSpace] is not None:
                frame = cv2.cvtColor(frame, COLOR_CHANGES[self.colourSpace])

            if self.resize is not None:
                 frame = cv2.resize(frame, self.resize)

            frames.append(frame)

        frames = np.array(frames)
        
        self.meanImage = np.mean(frames, axis=0) / 255
        self.stadImage = np.std(frames, axis=0) / 255


    def test(self):
        outputFolder = 'framesResult_adaptiveBGR'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            originalFrame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_COLOR)

            if COLOR_CHANGES[self.colourSpace] is not None:
                originalFrame = cv2.cvtColor(originalFrame, COLOR_CHANGES[self.colourSpace])

            if self.resize is not None:
                 originalFrame = cv2.resize(originalFrame, self.resize)
            originalFrame = originalFrame * (1. / 255)

            #Classification into foreground or background
            frame = np.abs(originalFrame - self.meanImage)
            result = frame >=  self.alpha * (self.stadImage + 2/255)

            result = np.any(result, axis=2)

            if self.morph:
                 result = result.astype(np.uint8)
                 result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_size)

            #ADAPTIVE PART -> if pixel is in the background, update mean and variance
            self.meanImage[result == 0] = self.ro*originalFrame[result == 0] + (1-self.ro)*self.meanImage[result == 0]
            self.stadImage[result == 0] = np.sqrt(self.ro*(originalFrame[result == 0]-self.meanImage[result == 0])**2 + (1-self.ro)*self.stadImage[result == 0]**2)

            result = result.astype(np.uint8) * 255
            
            #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  
            cv2.imwrite(outputFolder + '/frame' + str(i).zfill(5) + '.png', result)

    
    def fast_test(self, ro, alpha, colorSpace, kernel_size = None):

        self.alpha = alpha
        self.ro = ro
        self.kernel_size = kernel_size
        self.colourSpace = colorSpace

        outputFolder = 'framesResult_adaptiveBGR'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            originalFrame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_COLOR)

            if COLOR_CHANGES[self.colourSpace] is not None:
                originalFrame = cv2.cvtColor(originalFrame, COLOR_CHANGES[self.colourSpace])

            if self.resize is not None:
                 originalFrame = cv2.resize(originalFrame, self.resize)
            originalFrame = originalFrame * (1. / 255)

            #Classification into foreground or background
            frame = np.abs(originalFrame - self.meanImage)
            result = frame >=  self.alpha * (self.stadImage + 2/255)

            result = np.any(result, axis=2)

            if self.morph:
                 result = result.astype(np.uint8)
                 result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_size)

            #ADAPTIVE PART -> if pixel is in the background, update mean and variance
            self.meanImage[result == 0] = self.ro*originalFrame[result == 0] + (1-self.ro)*self.meanImage[result == 0]
            self.stadImage[result == 0] = np.sqrt(self.ro*(originalFrame[result == 0]-self.meanImage[result == 0])**2 + (1-self.ro)*self.stadImage[result == 0]**2)

            result = result.astype(np.uint8) * 255
            
            #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  
            cv2.imwrite(outputFolder + '/frame' + str(i).zfill(5) + '.png', result)


     
            

if __name__ == '__main__':
    # Read a video and save the frames on a folder
    inFrames = '/home/user/Documents/MASTER/C6/AICity_data/train/S03/c010/vdo.avi'
    outFrames = 'framesOriginal'
    modelo = BackgroundRemoval(inFrames, outFrames, morph = True, kernel_size=(11,11))

    # modelo.train_unoptimized()
    modelo.train()
    modelo.test()



    