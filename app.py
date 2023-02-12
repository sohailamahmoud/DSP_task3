from turtle import shape
from flask import Flask, render_template, request
import pickle
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np 
from utils import functions
from Speaker_Identification_Using_Machine_Learning import SpeakerIdentification

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/audioStreaming', methods=["POST"])
def audioStreaming():
    if request.method == 'POST':
        SpeakerIdentification.record_audio_test()
    
    speaker, mfcc,scores = SpeakerIdentification.test_model_speaker()
    word = SpeakerIdentification.test_model_word()

    if speaker == 0 :
        if word == 1:
            message = 'Hello Gehad. Door is opened successfully.'
        else :
            message = 'Hello Gehad. Password is incorrect.'
    elif speaker == 1 :
        if word == 1:
            message = 'Hello Rawan. Door is opened successfully.'
        else :
            message = 'Hello Rawan. Password is incorrect.'
    elif speaker == 2 :
        if word == 1:
            message = 'Hello Sohaila. Door is opened successfully.'
        else :
            message = 'Hello Sohaila. Password is incorrect.'
    else :
        message = "Not Recognized."


    functions.pltMFCC(speaker)
    return render_template("index.html", msg = message)
    

if __name__ == '__main__':
    app.run(debug=True, port=8000)

