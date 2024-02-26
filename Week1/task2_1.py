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

COLOR_CHANGES = {
    'grayscale' : cv2.COLOR_BGR2GRAY,
    'hsv' : cv2.COLOR_BGR2HSV,
    'readgs': cv2.IMREAD_GRAYSCALE
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

    def __init__(self, pathInput, pathOutput, colourSpace = "readgs", resize = None, alpha = 4, ro = 0.2, morph = False, kernel_size = (3,3)):
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
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])

            if self.resize is not None:
                 frame = cv2.resize(frame, self.resize)
            frames.append(frame)

        frames = np.array(frames)
        
        self.meanImage = np.mean(frames, axis=0) / 255
        self.stadImage = np.std(frames, axis=0) / 255


        

    def train_unoptimized(self, percentageOfFrames=0.25):
        # i still have to implement the size thing here
        lastFrame = math.floor(self.numFrames * percentageOfFrames)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = None
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])

            frame = (1./255) * frame
            
            if frames is None:
                frames = np.zeros_like(frame)

            frames += frame

        frames = np.array(frames)

        self.meanImage = frames / (lastFrame + 1)
        #cv2.imwrite('mean_image.png', self.meanImage*255)


        frames = None
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])

            frame = (1./255) * frame
            
            if frames is None:
                frames = np.zeros_like(frame)

            frames += (frame - self.meanImage) ** 2

        frames = (1. / (lastFrame + 1)) * frames
        frames = np.sqrt(frames)

        self.stadImage = frames

        """#cv2.imwrite('stad.png', self.stadImage*255)
        sns.heatmap(self.stadImage, cmap='viridis')

        # Save the heatmap as an image
        plt.savefig('stad.png')"""


    def test(self):

        outputFolder = 'framesResult_adaptive'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])
            if self.resize is not None:
                 frame = cv2.resize(frame, self.resize)
            frame = frame * (1. / 255)

            frame = np.abs(frame - self.meanImage)
            result = frame >=  self.alpha * (self.stadImage + 2/255)
            


            if self.morph:
                 result = result.astype(np.uint8)
                 result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_size)

            #ADAPTIVE PART
            #if background adapt alpha and stad
            self.meanImage[result == 0] = self.ro*result[result == 0] + (1-self.ro)*self.meanImage[result == 0]
            self.stadImage[result == 0] = self.ro*(result[result == 0]-self.meanImage[result == 0])**2 + (1-self.ro)*self.stadImage[result == 0]**2

            result = result.astype(np.uint8) * 255
            
            #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  
            cv2.imwrite(outputFolder + '/frame' + str(i).zfill(5) + '.png', result)

    def test_updated(self):
        outputFolder = 'framesResult_adaptive'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            originalFrame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])
            if self.resize is not None:
                 originalFrame = cv2.resize(originalFrame, self.resize)
            originalFrame = originalFrame * (1. / 255)

            frame = np.abs(originalFrame - self.meanImage)
            result = frame >=  self.alpha * (self.stadImage + 2/255)

            if self.morph:
                 result = result.astype(np.uint8)
                 result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self.kernel_size)

            #ADAPTIVE PART
            #if background adapt alpha and stad
            self.meanImage[result == 0] = self.ro*originalFrame[result == 0] + (1-self.ro)*self.meanImage[result == 0]
            self.stadImage[result == 0] = self.ro*(originalFrame[result == 0]-self.meanImage[result == 0])**2 + (1-self.ro)*self.stadImage[result == 0]**2

            result = result.astype(np.uint8) * 255

            cv2.imshow('frame', self.meanImage)
            cv2.waitKey(1)
            
            #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  
            cv2.imwrite(outputFolder + '/frame' + str(i).zfill(5) + '.png', result)


     
            

if __name__ == '__main__':
    # Read a video and save the frames on a folder
    inFrames = '/home/user/Documents/MASTER/C6/AICity_data/train/S03/c010/vdo.avi'
    outFrames = 'framesOriginal'
    modelo = BackgroundRemoval(inFrames, outFrames, morph = True, kernel_size=(3,3))

    modelo.train_unoptimized()
    #modelo.train()
    modelo.test()



    