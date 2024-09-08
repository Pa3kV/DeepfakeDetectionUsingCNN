import tensorflow as tf
import os
import cv2
import glob
from mtcnn import MTCNN

def extractFramesFromVideos(video_path, image_size, num_frames=10):
    frames = []
    face_detector = MTCNN()

    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < (num_frames * 3):
        return frames

    for frame_num in range(num_frames * 3):
        if frame_num % 3 == 0:
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detectedFaces = face_detector.detect_faces(frame)
            if len(detectedFaces) > 0:
                x, y, width, height = detectedFaces[0]['box']

                if x < 0: x = 0
                if y < 0: y = 0
                if x + width > frame.shape[1]: width = frame.shape[1] - x
                if y + height > frame.shape[0]: height = frame.shape[0] - y

                faceImg = frame[y:y+height, x:x+width]
                faceImg = cv2.resize(faceImg, image_size)
                frames.append(faceImg)

    capture.release()
    return frames


dataset_root = os.path.join(os.getcwd() + '/sample_videos')
srcTrainReal = dataset_root + '/train/Real'
srcTrainFake = dataset_root + '/train/Fake'
srcTestReal = dataset_root + '/test/Real'
srcTestFake = dataset_root + '/test/Fake'

framesRoot = os.path.join(os.getcwd() + '/frames')
framesTrainReal = framesRoot + '/train/Real'
framesTrainFake = framesRoot + '/train/Fake'
framesTestReal = framesRoot + '/test/Real'
framesTestFake = framesRoot + '/test/Fake'

imageSize = (224,224)
numOfFramesPerVideo = 10

realVideoTrainPaths = glob.glob(os.path.join(srcTrainReal, "*.mp4"))
fakeVideoTrainPaths = glob.glob(os.path.join(srcTrainFake, "*.mp4"))
realVideoTestPaths = glob.glob(os.path.join(srcTestReal, "*.mp4"))
fakeVideoTestPaths = glob.glob(os.path.join(srcTestFake, "*.mp4"))

os.makedirs(framesTrainReal)
os.makedirs(framesTrainFake)
os.makedirs(framesTestReal)
os.makedirs(framesTestFake)

for path in realVideoTrainPaths:
    frames = extractFramesFromVideos(path, imageSize, numOfFramesPerVideo)
    if frames != []:
        for frame in frames:
            fileCount = len(os.listdir(framesTrainReal))
            fileName = f'frame_{fileCount + 1}.jpg'
            cv2.imwrite(os.path.join(framesTrainReal, fileName), frame) 

for path in fakeVideoTrainPaths:
    frames = extractFramesFromVideos(path, imageSize, numOfFramesPerVideo)
    if frames != []:
        for frame in frames:
            fileCount = len(os.listdir(framesTrainFake))
            fileName = f'frame_{fileCount + 1}.jpg'
            cv2.imwrite(os.path.join(framesTrainFake, fileName), frame)

for path in realVideoTestPaths:
    frames = extractFramesFromVideos(path, imageSize, numOfFramesPerVideo)
    if frames != []:
        for frame in frames:
            fileCount = len(os.listdir(framesTestReal))
            fileName = f'frame_{fileCount + 1}.jpg'
            cv2.imwrite(os.path.join(framesTestReal, fileName), frame)

for path in fakeVideoTestPaths:
    frames = extractFramesFromVideos(path, imageSize, numOfFramesPerVideo)
    if frames != []:
        for frame in frames:
            fileCount = len(os.listdir(framesTestFake))
            fileName = f'frame_{fileCount + 1}.jpg'
            cv2.imwrite(os.path.join(framesTestFake, fileName), frame)


print("Frames copied successfully")