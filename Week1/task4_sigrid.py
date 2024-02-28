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
    'readgs': cv2.IMREAD_GRAYSCALE,
    'bgr': cv2.IMREAD_COLOR,
    'lab': cv2.COLOR_BGR2Lab,
    'yuv': cv2.COLOR_BGR2YUV,

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
            file_list = sorted(os.listdir(pathOutput))
            return totalFrames, file_list
        
        else:
            print('The folder already exists, reading the content.')
            file_list = sorted([name for name in os.listdir(pathOutput) if os.path.isfile(os.path.join(pathOutput, name))])
            return len(file_list), file_list

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

    def __init__(self, pathInput, pathOutput, colourSpace = "readgs", resize = None, alpha = 4, ro = 0.01, morph = False, kernel_size = (3,3), channels = "3", gaussians = 1, T = 1):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.colourSpace = colourSpace
        self.resize = resize
        self.alpha = alpha
        self.ro = ro
        self.morph = morph
        self.kernel_size = kernel_size
        self.channels = channels
        self.gaussians = gaussians
        self.T = T
        self.numFrames, self.listFrames = getFrames(self.framesInPath, self.framesOutPath)
        
        
    def getNumFrames(self):
        return self.numFrames
    
    def getFrames(self):
        return self.listFrames
    
    def get_channels(self, frame):
        a,b, c = cv2.split(frame)

        if self.channels == "1a":
            return a
        if self.channels == "1b":
            return b
        if self.channels == "1c":
            return c
        if self.channels == "1-2":
            return cv2.merge((a, b))
        if self.channels == "2-3":
            return cv2.merge((b,c))
        if self.channels == "1-3":
            return cv2.merge((a,c))

        print("something wrong with the channels")
        return frame

    
    def train(self, percentageOfFrames=0.25):
        # REMAINS THE SAME AS IN TASK1.1
        lastFrame = math.floor(self.numFrames * percentageOfFrames)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = []
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])
            
            if self.channels != 3 and self.colourSpace != 'readgs':
                frame = self.get_channels(frame)

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

            if self.channels != 3 and self.colourSpace != 'readgs':
                frame = self.get_channels(frame)

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
            if self.channels != 3 and self.colourSpace != 'readgs':
                frame = self.get_channels(frame)

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
        outputFolder = 'framesResult_adaptiveHSV'
        if os.path.exists(outputFolder):  
                shutil.rmtree(outputFolder)

        os.mkdir(outputFolder)

        for i in tqdm(range(0, self.numFrames)):
            originalFrame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], COLOR_CHANGES[self.colourSpace])
            if self.channels != 3 and self.colourSpace != 'readgs':
                originalFrame = self.get_channels(originalFrame)
            if self.resize is not None:
                 originalFrame = cv2.resize(originalFrame, self.resize)
            originalFrame = originalFrame * (1. / 255)

            #Classification into foreground or background
            frame = np.abs(originalFrame - self.meanImage)
            result = frame >=  self.alpha * (self.stadImage + 2/255)

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
    """
    COSES A TENIR EN COMPTE:
    - crec que el train es queda igual?
    - falta fer lo del gmm
    - cal optimitzar per:
        - ro: de 0 a nose quant
        - alpha: de 0 a 1
        - colorspace: els que he definit
        - channels: 1a, 1b, 1c, 1-2, 1-3, 2-3, 3 (if only 2 channels, cant be saved as image)
        - number of gaussians: 1- k
        - T threshold to say if belonds to background or foreground (entre 0 i 1?)


    """


    # Read a video and save the frames on a folder
    inFrames = '/home/user/Documents/MASTER/C6/AICity_data/train/S03/c010/vdo.avi'
    outFrames = 'framesOriginal'
    modelo = BackgroundRemoval(inFrames, outFrames, morph = True, kernel_size=(11,11), colourSpace = 'hsv', channels = "1-2", gaussians = 7, T = 0.7)

    modelo.train_unoptimized()
    #modelo.train()
    modelo.test()



    