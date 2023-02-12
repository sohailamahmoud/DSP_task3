import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from Models_Notebook import *

warnings.filterwarnings("ignore")

def calculate_delta(array):

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined, mfcc_feature



def record_audio_test():
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 44100
	CHUNK = 512
	RECORD_SECONDS = 2.5
	device_index = 2
	audio = pyaudio.PyAudio()

	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,input_device_index = 1,
						frames_per_buffer=CHUNK)
	print ("recording started")
	Recordframes = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			Recordframes.append(data)
	print ("recording stopped")
	stream.stop_stream()
	stream.close()
	audio.terminate()

	OUTPUT_FILENAME="sample.wav"
	WAVE_OUTPUT_FILENAME=os.path.join("D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\testing_set",OUTPUT_FILENAME)
	trainedfilelist = open("testing_set_addition.txt", 'a')
	trainedfilelist.write(OUTPUT_FILENAME+"\n")
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(Recordframes))
	waveFile.close()


def test_model_speaker():

	source		= "testing_set\\"  
	modelpath 	= "D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\trained_models\Speaker\\"
	test_file 	= "testing_set_addition.txt"       
	file_paths 	= open(test_file,'r')

	gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

	#Load the Gaussian gender Models
	models     = [pickle.load(open(fname,'rb')) for fname in gmm_files]
	speakers   = [fname.split("\\")[-1].split(".gmm")[0]  for fname in gmm_files]
	speakers.append("Others")
	
	# Read the test directory and get the list of test audio files 
	for path in file_paths:   

		path = path.strip()   
		sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\testing_set\sample.wav')
		vector, mfcc   = extract_features(audio,sr)

		log_likelihood = np.zeros(len(models)) 

		for i in range(len(models)):
			gmm    = models[i]  #checking with each model one by one
			scores = np.array(gmm.score(vector))
			log_likelihood[i] = scores.sum()

		winner = np.argmax(log_likelihood)

		flag = False
		flagList = log_likelihood - max(log_likelihood)
		for i in range (len(flagList)) :
			if flagList[i] == 0 :
				continue
			if abs(flagList[i]) < 0.4 :
				flag = True

		if flag :
			winner = 3
	
		print("\tdetected as - ", speakers[winner])
		time.sleep(1.0)  
	print(log_likelihood)
	return winner, mfcc,log_likelihood


def test_model_word():

	source		= "testing_set\\"  
	modelpath 	= "D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\trained_models\Word\\"
	test_file 	= "testing_set_addition.txt"       
	file_paths 	= open(test_file,'r')

	gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

	#Load the Gaussian gender Models
	models     = [pickle.load(open(fname,'rb')) for fname in gmm_files]
	speakers   = [fname.split("\\")[-1].split(".gmm")[0]  for fname in gmm_files]
	speakers.append("Others")
	
	# Read the test directory and get the list of test audio files 
	for path in file_paths:   

		path = path.strip()   
		sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\testing_set\sample.wav')
		vector, mfcc   = extract_features(audio,sr)

		log_likelihood = np.zeros(len(models)) 

		for i in range(len(models)):
			gmm    = models[i]  #checking with each model one by one
			scores = np.array(gmm.score(vector))
			log_likelihood[i] = scores.sum()
		winner = 0
		winner = np.argmax(log_likelihood)

		print("\tdetected as - ", speakers[winner])
		time.sleep(1.0)  
	return winner


# while True:
# 	choice=int(input("\n 1.Record audio for testing \n 2.Test Model\n"))
# 	if(choice==1):
# 		record_audio_test()
# 	elif(choice==2):
# 		test_model_speaker()
# 		test_model_word()
# 	if(choice>4):
# 		exit()

