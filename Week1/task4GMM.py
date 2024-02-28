from re import sub
import cv2
import os
import shutil
from sympy import Subs

from tqdm import tqdm
import math
import numpy as np

from numpy.linalg import norm, inv

from scipy.stats import multivariate_normal as mv_norm

'''
We get inspiration for our implementation using the following repository: https://github.com/ZiyangS/BackgroundSubtraction/blob/master/GMM.py
'''

init_means = np.zeros(3)
# init_var = np.array([255, 255, 255])
init_var = 225 * np.eye(3)

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

class GMM:
    def __init__(self, pathInput, pathOutput, colourSpace = "readgs", resize = None, alpha = 4, ro = 0.01, morph = False, kernel_size = (3,3), channels = "3", gaussians = 3, T = 0.9):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.colourSpace = colourSpace
        self.resize = resize
        self.alpha = alpha
        self.ro = ro
        self.morph = morph
        self.kernel_size = kernel_size
        self.numFrames, self.listFrames = getFrames(self.framesInPath, self.framesOutPath)
        self.channels = channels
        self.gaussians = gaussians
        self.T = T
        self.B = None

        self.init_weight = np.random.rand(self.gaussians)
        self.init_weight /= self.init_weight.sum()

    def check1(self, pixel, mu, sigma):

        print('Pixel: ', pixel)
        print('Mu: ', mu)
        print('Sigma: ', sigma)
        
        substraction = abs(pixel - mu)
        comparasion = self.alpha * (sigma + 2 / 255)

        if substraction >= comparasion:
            return False
        else:
            return True
        
    def check(self, pixel, mu, sigma):
        '''
        check whether a pixel match a Gaussian distribution. Matching means pixel is less than
        2.5 standard deviations away from a Gaussian distribution.
        '''
        x = np.mat(np.reshape(pixel, (3, 1)))
        u = np.mat(mu).T
        sigma = np.mat(sigma)
        # calculate Mahalanobis distance
        d = np.sqrt((x-u).T*sigma.I*(x-u))
        if d < 2.5:
            return True
        else:
            return False
        
    def reorder(self):
        '''
        reorder the estimated components based on the ratio pi / the norm of standard deviation.
        the first B components are chosen as background components
        the default threshold is 0.75
        '''
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                k_weight = self.weight[i][j]
                k_norm = np.array([norm(np.sqrt(self.sigma[i][j][k])) for k in range(self.gaussians)])
                ratio = k_weight/k_norm
                descending_order = np.argsort(-ratio)

                self.weight[i][j] = self.weight[i][j][descending_order]
                self.means[i][j] = self.means[i][j][descending_order]
                self.sigma[i][j] = self.sigma[i][j][descending_order]

                cum_weight = 0
                for index, order in enumerate(descending_order):
                    cum_weight += self.weight[i][j][index]
                    if cum_weight > self.T:
                        self.B[i][j] = index + 1
                        break


    def train(self, percentageOfFrames = 0.25):
        # REMAINS THE SAME AS IN TASK1.1
        lastFrame = math.floor(self.numFrames * percentageOfFrames)
        
        img_shape = cv2.imread(self.framesOutPath + '/' + self.listFrames[0]).shape

        self.img_shape = img_shape

        print(img_shape)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = []
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i])
            if self.resize != None:
                frame = cv2.resize(frame, self.resize)
            frames.append(frame)

        # Per cada pixel, per cada canal, per k gaussians
        self.means = np.array([[[init_means.copy() for _ in range(self.gaussians)] for _ in range(img_shape[1])]
                             for _ in range(img_shape[0])])
        
        self.sigma = np.array([[[init_var.copy() for k in range(self.gaussians)] for j in range(img_shape[1])]
                             for i in range(img_shape[0])])
        
        self.weight = np.array([[self.init_weight.copy() for j in range(self.img_shape[1])] for i in range(img_shape[0])])

        self.B = np.ones(self.img_shape[0:2], dtype=int)

        initial_frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[0])
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                for k in range(self.gaussians):
                    self.means[i][j][k] = np.array(initial_frame[i][j]).reshape(1,3)

        for file in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[file])
            if self.resize != None:
                frame = cv2.resize(frame, self.resize)

            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    match = -1
                    for k in range(self.gaussians):
                        if self.check(frame[i][j], self.means[i][j][k], self.sigma[i][j][k]):
                            match = k
                            break

                    if match != -1:
                        mu = self.means[i][j][k]
                        sigma = self.sigma[i][j][k]
                        x = frame[i][j].astype(float)
                        delta = x - mu
                        rho = self.alpha * mv_norm.pdf(frame[i][j], mu, sigma)
                        self.weight[i][j] = (1 - self.alpha) * self.weight[i][j]
                        self.weight[i][j][match] += self.alpha
                        self.means[i][j][k] = mu + rho * delta
                        self.sigma[i][j][k] = sigma + rho * (np.matmul(delta, delta.T) - sigma)
                    
                    if match == -1:
                        w_list = [self.weight[i][j][k] for k in range(self.gaussians)]
                        id = w_list.index(min(w_list))
                        self.means[i][j][id] = np.array(frame[i][j]).reshape(1, 3)
                        self.sigma[i][j][id] = np.array(init_var)

        self.reorder()

    def test(self, img):
        '''
        infer whether its background or foregound
        if the pixel is background, both values of rgb will set to 255. Otherwise not change the value
        '''
        result = np.array(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(self.B[i][j]):
                    if self.check(img[i][j], self.means[i][j][k], self.sigma[i][j][k]):
                        # [255, 255, 255] is white, the background color will be set to white
                        result[i][j] = [255, 255, 255]
                        break
        return result
    
    def train_optimized(self, percentageOfFrames=0.25):
        lastFrame = math.floor(self.numFrames * percentageOfFrames)
        img_shape = cv2.imread(self.framesOutPath + '/' + self.listFrames[0]).shape
        self.img_shape = img_shape

        frames = []
        for i in tqdm(range(lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i])
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)
            frames.append(frame)

        self.means = np.array([[[init_means.copy() for _ in range(self.gaussians)] for _ in range(img_shape[1])]
                            for _ in range(img_shape[0])])
        self.sigma = np.array([[[init_var.copy() for _ in range(self.gaussians)] for _ in range(img_shape[1])]
                            for _ in range(img_shape[0])])
        self.weight = np.array([[[self.init_weight.copy() for _ in range(self.gaussians)] for _ in range(img_shape[1])]
                                for _ in range(img_shape[0])])
        self.B = np.ones(self.img_shape[:2], dtype=int)

        initial_frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[0])
        self.means[:, :, :] = np.array(initial_frame).reshape(img_shape[0], img_shape[1], 1, 3)

        for file in tqdm(range(lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[file])
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            match = np.zeros((frame.shape[0], frame.shape[1]), dtype=int) - 1
            for k in range(self.gaussians):
                mask = match == -1
                if not mask.any():
                    break
                if k > 0:
                    mask &= self.check(frame, self.means[:, :, k], self.sigma[:, :, k])
                match[mask] = k

            delta = frame.astype(float) - self.means[:, :, match].reshape(frame.shape[0], frame.shape[1], 3)
            rho = self.alpha * mv_norm.pdf(frame, self.means[:, :, match].reshape(frame.shape[0], frame.shape[1], 3),
                                            self.sigma[:, :, match].reshape(frame.shape[0], frame.shape[1], 3))

            self.weight = (1 - self.alpha) * self.weight
            self.weight[range(frame.shape[0])[:, None], range(frame.shape[1]), match] += self.alpha
            self.means[:, :, match] += rho[:, :, :, None] * delta[:, :, :, None]
            self.sigma[:, :, match] += rho[:, :, :, None] * (np.matmul(delta[:, :, :, None], delta[:, :, :, None].transpose(0, 1, 3, 2)) - self.sigma[:, :, match])

        self.reorder()
    
    def test_optimized(self, img):
        result = np.array(img)
        match = np.ones((img.shape[0], img.shape[1]), dtype=bool)

        for k in range(self.gaussians):
            match &= self.check(img, self.means[:, :, k], self.sigma[:, :, k])

        result[match] = [255, 255, 255]
        return result
    

        


if __name__  == '__main__':
    inFrames = 'c010/vdo.avi'
    outFrames = 'framesOriginal'

    kernel_number = (11, 11)
    modelo = GMM(inFrames, outFrames, alpha = 0.25, morph = False, resize=(100, 100), kernel_size=(kernel_number, kernel_number))
    modelo.train()

    frameNumber = 600
    frameActual = cv2.imread(outFrames + '/' + modelo.listFrames[frameNumber])
    result = modelo.test(frameActual)

    cv2.imshow('result', result)
    cv2.waitKey(0)

        