import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import python_speech_features as mfcc
import numpy.random as random
from scipy.io.wavfile import read




def mfcc_spectro(mfccs):
    fig = plt.figure(facecolor='#C5E5EC')
    sns.heatmap(mfccs.T)
    plt.xlabel("time", {'color': '#104374'})
    plt.yticks(color = '#104374')
    plt.xticks(color = '#104374')
    plt.ylabel("frequency", {'color': '#104374'})
    plt.title("MFCC", color = '#104374')
    plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\mfcc_plot.png')




def plotScatter(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out):
    fig = plt.figure(facecolor='#C5E5EC')
    plt.scatter(mfcc_g[13], mfcc_g[17], color = 'red', marker='x', label = 'Gehad')
    plt.scatter(mfcc_r[13] + 400, mfcc_r[17], color = 'blue', marker='x', label = 'Rawan')
    plt.scatter(mfcc_s[13] + 900, mfcc_s[17], color = 'green', marker='x', label = 'Sohaila')
    if speaker == 0:
        plt.scatter(mfcc_out[13], mfcc_out[17], color = 'black', marker = 'o', label = 'Output')
    elif speaker == 1:
        plt.scatter(mfcc_out[13] + 400, mfcc_out[17], color = 'black', marker = 'o', label = 'Output')
    elif speaker == 2:
        plt.scatter(mfcc_out[13] + 900, mfcc_out[17], color = 'black', marker = 'o', label = 'Output')
    else:
        plt.scatter(mfcc_out[13] + 1200, mfcc_out[17], color = 'black', marker = 'o', label = 'Output')
    plt.legend()
    plt.xlim(-250, 1600)
    plt.xlabel('MFCC13', color = '#104374')
    plt.ylabel('MFCC17', color = '#104374')
    plt.yticks(color = '#104374')
    plt.xticks(color = '#104374')
    plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\mfcc_scatter.png')


def plot(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out):
    fig = plt.figure(facecolor='#C5E5EC')
    plt.plot(mfcc_g[13], color = 'red', label = 'Gehad')
    plt.plot(mfcc_r[13] + 400, color = 'blue', label = 'Rawan')
    plt.plot(mfcc_s[13] + 900, color = 'green', label = 'Sohaila')
    if speaker == 0:
        plt.plot(mfcc_out[13], color = 'black', label = 'Output')
    elif speaker == 1:
        plt.plot(mfcc_out[13] + 400, color = 'black', label = 'Output')
    elif speaker == 2:
        plt.plot(mfcc_out[13] + 900, color = 'black', label = 'Output')
    else:
        plt.plot(mfcc_out[13] + 1200, color = 'black', label = 'Output')
    plt.legend()
    plt.title('MFCC13', color = '#104374')
    plt.ylim(-250, 1600)
    plt.yticks(color = '#104374')
    plt.xticks(color = '#104374')
    plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\mfcc_plot.png')


def mfcc_dist(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out):
    plt.style.use('ggplot')
    fig = plt.figure(facecolor='#C5E5EC')
    sns.distplot(mfcc_g, color = 'red', hist=False, label= 'Gehad')
    sns.distplot(mfcc_r +50, color = 'blue', hist=False, label= 'Rawan')
    sns.distplot(mfcc_s+100, color = 'green', hist=False,label= 'Sohaila')
    if speaker == 0:
        sns.distplot(mfcc_out, color = 'black', hist=False, label= 'Output')
    elif speaker == 1:
        sns.distplot(mfcc_out+50, color = 'black', hist=False, label= 'Output')
    elif speaker == 2:
        sns.distplot(mfcc_out+100, color = 'black', hist=False, label= 'Output')
    else:
        sns.distplot(mfcc_out+150, color = 'black', hist=False, label= 'Output')
    
    plt.legend()
    # plt.ylim(-5,5)
    # plt.xlim(-20, 20)
    plt.xlabel("MFCC", {'color': '#104374'})
    plt.yticks(color = '#104374')
    plt.xticks(color = '#104374')
    plt.title('Density Plot of MFCC', color = '#104374')
    plt.tight_layout()
    plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\mfcc_dist.png')


def pltMFCC(speaker):
    sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\GehadOpen\Gehad(1).wav')
    mfcc_g = mfcc.mfcc(audio, sr, nfft=40)

    sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\RawanOpen\Rawan(1).wav')
    mfcc_r = mfcc.mfcc(audio, sr, nfft=40)

    sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\SohailaOpen\Sohaila(1).wav')
    mfcc_s = mfcc.mfcc(audio, sr, nfft=40)

    sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\testing_set\sample.wav')
    mfcc_out = mfcc.mfcc(audio, sr, nfft=40)

    plotScatter(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out)
    plot(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out)
    mfcc_dist(speaker, mfcc_g, mfcc_r, mfcc_s, mfcc_out)






###

# def plotout(plotting):
#     mfccplotx = random.uniform((-13),(-15))
#     if plotting == 0:
#         mfccplot = random.uniform((-11.9),(-12.2))
#     elif plotting == 1:
#         mfccplot = random.uniform((-11.2),(-11.8))
#     elif plotting == 2:
#         mfccplot = random.uniform((-12.3),(-13))
#     else:
#         mfccplot = random.uniform((-8),(-11))
#     return mfccplotx, mfccplot

# def listmfcc(mfcclist):
#     mfcc_mean_list = []
#     for i in mfcclist:
#         mfcc_mean_list.append(np.mean(i))
#     mfcc_20 = mfcc_mean_list[:20]
#     return mfcc_20


# def plotMfccSpeaker(file_name, plotting):
#     mfcceoutx, mfcceout = plotout(plotting)
#     fig=plt.figure(figsize=(5,5), facecolor="#C5E5EC")
#     audio, sfreq = librosa.load(file_name)
#     mfccout = mfcc.mfcc(audio,sfreq,nfft=20)
#     audioGehad,sfreqGehad = librosa.load('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\GehadOpen\Gehad(1).wav')
#     mfccGehad = mfcc.mfcc(audioGehad, sfreqGehad,nfft=20)
#     hplot(mfccGehad,0)
#     audioRawan,sfreqRawan = librosa.load('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\RawanOpen\Rawan(1).wav')
#     mfccRawan = mfcc.mfcc(audioRawan, sfreqRawan,nfft=20)
#     hplot(mfccRawan,1)
#     audioSohaila,sfreqSohaila = librosa.load('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\SohailaOpen\Sohaila(1).wav')
#     mfccSohaila = mfcc.mfcc(audioSohaila, sfreqSohaila,nfft=20)
#     hplot(mfccSohaila,2)
#     plt.title('MFCC')
#     mfcc_20 = listmfcc(mfccout)
#     mfcc_20Gehad = listmfcc(mfccGehad)
#     mfcc_20Rawan = listmfcc(mfccRawan)
#     mfcc_20Sohaila = listmfcc(mfccSohaila)
#     mfccoutx=[mfcc_20[13]]
#     mfccout=[mfcc_20[17]]
#     plt.scatter(x=mfcceoutx , y=mfcceout,color = 'purple',label='output')
#     plt.axhline(y=mfcc_20Sohaila[13],color ='orange',label ='Sohaila')
#     plt.axhline(y= mfcc_20Rawan[13],color = 'blue',label='Rawan')
#     plt.axhline(y= mfcc_20Gehad[13],color = 'red',label = 'Gehad')
#     plt.xlabel('MFCC')
#     # plt.ylabel('MFCC13')
#     # plt.scatter(x = coeff ,y = mfcc_20Habiba,color ='red',label ='Habiba')
#     plt.legend(loc ='upper right')
#     plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\speaker.png')

#     mfccImg = True
#     return mfccImg


# def distplot(mfccg, mfccr, mfccs, mfcco):
#     plt.style.use('ggplot')
#     fig = plt.figure(facecolor='#C5E5EC')
#     plt.xlabel("MFCC", {'color': '#104374'})
#     plt.yticks(color = '#104374')
#     plt.xticks(color = '#104374')
#     sns.distplot(mfccg, color = 'blue', hist=False)
#     sns.distplot(mfccr, color = 'red', hist=False)
#     sns.distplot(mfccs, color = 'green', hist=False)
#     sns.distplot(mfcco, color = 'black', hist=False)
#     plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\dist.png')
    # if speaker == 0:
    #     sns.distplot(mfcc, color = 'blue', hist=False)
    #     plt.title('Gehad', color = '#104374')
    #     plt.tight_layout()
    #     plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\gehad_dist.png')
    # elif speaker == 1:
    #     sns.distplot(mfcc, color = 'red', hist=False)
    #     plt.title('Rawan', color = '#104374')
    #     plt.tight_layout()
    #     plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\\rawan_dist.png')
    # elif speaker == 2:
    #     sns.distplot(mfcc, color = 'green', hist=False)
    #     plt.title('Sohaila', color = '#104374')
    #     plt.tight_layout()
    #     plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\sohaila_dist.png')

# def plotAudio(speaker):
#     sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\GehadOpen\Gehad(1).wav')
#     mfccg = mfcc.mfcc(audio, sr, nfft=20)

#     sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\RawanOpen\Rawan(1).wav')
#     mfccr = mfcc.mfcc(audio, sr, nfft=20)

#     sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Models_Notebook\SohailaOpen\Sohaila(1).wav')
#     mfccs = mfcc.mfcc(audio, sr, nfft=20)

#     sr,audio = read('D:\Edu\DSP\Tasks\Task3\\task3_gmm\Speaker_Identification_Using_Machine_Learning\\testing_set\sample.wav')
#     mfcco = mfcc.mfcc(audio, sr, nfft=20)

#     # plt.scatter(mfccg[13],mfccg[5], color = 'red',label='Gehad')
#     # plt.scatter(mfccr[13],mfccr[5],color = 'blue',label='Rawan')
#     # plt.scatter(mfccs[13],mfccs[5],color = 'green',label='Sohaila')
#     # # plt.scatter(x=mfcco[] , y=mfcceout,color = 'purple',label='output')

#     # plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\scatter.png')

#     plt.style.use('ggplot')
#     fig = plt.figure(facecolor='#C5E5EC')
#     plt.xlabel("MFCC", {'color': '#104374'})
#     plt.yticks(color = '#104374')
#     plt.xticks(color = '#104374')
#     if speaker == 0:
#         ax =sns.distplot(mfccg, color = 'blue', hist=False)
#         x = ax.lines[0].get_xdata() # Get the x data of the distribution
#         y = ax.lines[0].get_ydata() # Get the y data of the distribution
#         ax.set_xlim(-150,150)
#         maxid = np.argmax(y)
#         print(x[maxid])
#         plt.axvline(x=x[maxid] - 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] - 10))[:5] , xy= (x[maxid] - 0.8,0.4))
#         plt.axvline(x=x[maxid] + 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] + 10))[:5] , xy= (x[maxid] + 0.3,0.4))
#         plt.title('Gehad', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\gehad_dist.png')
#     elif speaker == 1:
#         ax = sns.distplot(mfccr, color = 'red', hist=False)
#         x = ax.lines[0].get_xdata() # Get the x data of the distribution
#         y = ax.lines[0].get_ydata() # Get the y data of the distribution
#         ax.set_xlim(-150,150)
#         maxid = np.argmax(y)
#         print(x[maxid])
#         plt.axvline(x=x[maxid] - 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] - 10))[:5] , xy= (x[maxid] - 0.8,0.4))
#         plt.axvline(x=x[maxid] + 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] + 10))[:5] , xy= (x[maxid] + 0.3,0.4))
#         plt.title('Rawan', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\\rawan_dist.png')
#     elif speaker == 2:
#         ax = sns.distplot(mfccs, color = 'green', hist=False)
#         x = ax.lines[0].get_xdata() # Get the x data of the distribution
#         y = ax.lines[0].get_ydata() # Get the y data of the distribution
#         ax.set_xlim(-150,150)
#         maxid = np.argmax(y)
#         print(x[maxid])
#         plt.axvline(x=x[maxid] - 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] - 10))[:5] , xy= (x[maxid] - 0.8,0.4))
#         plt.axvline(x=x[maxid] + 10, color='b', linestyle='--')
#         ax.annotate(text = str((x[maxid] + 10))[:5] , xy= (x[maxid] + 0.3,0.4))
#         plt.title('Sohaila', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\sohaila_dist.png')


# def hplot(mfcc , speaker):
#     plt.style.use('ggplot')
#     fig = plt.figure(facecolor='#C5E5EC')
#     plt.plot(mfcc[13])
#     if speaker == 0:
#         plt.title('Gehad', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\gehad_dist.png')
#     elif speaker == 1:
#         plt.title('Rawan', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\\rawan_dist.png')
#     elif speaker == 2:
#         plt.title('Sohaila', color = '#104374')
#         plt.tight_layout()
#         plt.savefig('D:\Edu\DSP\Tasks\Task3\\task3_gmm\static\img\sohaila_dist.png')


