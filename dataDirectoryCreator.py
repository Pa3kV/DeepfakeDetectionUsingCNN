import os
import shutil
import random

def copyFilesToDstFolder(sourceFolder, destinationFolder, sampleNumber=200):
    videoList = os.listdir(sourceFolder)
    for file in random.sample(videoList, sampleNumber):
        srcFile = os.path.join(sourceFolder, file)
        dstFile = os.path.join(destinationFolder, file)
        shutil.copy(srcFile, dstFile)

def moveTestFiles(dstFile, isReal, sampleNumber = 40):
    currDir = os.getcwd()
    fileName = 'List_of_testing_videos.txt'
    listFilePath = os.path.join(currDir, fileName)
    with open(listFilePath, 'r') as file:
        lines = file.readlines()

    selectedVideos = []
    for line in lines:
        if len(selectedVideos) >= sampleNumber:
            break
        line = line.strip()
        parts = line.split()
        label = "1" if isReal else "0"
        if parts[0] == label:
            selectedVideos.append(parts[1])

    for video in selectedVideos:
        filePath = os.path.join(currDir, "input", video)
        shutil.move(filePath, dstFile)


sourceReal = os.path.join(os.getcwd() + '/input/Celeb-real')
sourceFake = os.path.join(os.getcwd() + '/input/Celeb-synthesis')
dataset_root = os.path.join(os.getcwd() + '/sample_videos')
destinationReal = dataset_root + '/train/Real'
destinationFake = dataset_root + '/train/Fake'
testReal = dataset_root + '/test/Real'
testFake = dataset_root + '/test/Fake'

realDstFolderCreated = True
fakeDstFolderCreated = True
testRealDirCreated = True
testFakeDirCreated = True

samplesNum = 200
testSamplesNum = samplesNum * 0.2

try:
    os.makedirs(destinationReal)
except:
    realDstFolderCreated = False

try:
    os.makedirs(destinationFake)
except:
    fakeDstFolderCreated = False

try:
    os.makedirs(testReal)
except:
    testRealDirCreated = False

try:
    os.makedirs(testFake)
except:
    testFakeDirCreated = False


if testRealDirCreated:
    moveTestFiles(testReal, True, testSamplesNum)
if testFakeDirCreated:
    moveTestFiles(testFake, False, testSamplesNum)

if realDstFolderCreated:
    copyFilesToDstFolder(sourceReal, destinationReal, samplesNum)
if fakeDstFolderCreated:
    copyFilesToDstFolder(sourceFake, destinationFake, samplesNum)

print("Videos copied successfully")