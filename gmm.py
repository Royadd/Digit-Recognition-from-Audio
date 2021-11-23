#from future import print_function
import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from python_speech_features import mfcc
from sklearn.metrics import confusion_matrix as cm
from sklearn.preprocessing import scale
import librosa as lib
from librosa.feature import delta

def extract_features(audio,fs):
    coeff=mfcc(audio,fs,winlen=0.01, winstep=0.003, numcep=13)
    Delta=delta(coeff,order=1)
    deltaDelta=delta(coeff,order=2)
    features=np.hstack((coeff,Delta,deltaDelta))
    return features

n=10

train_directory='/media/adr/WORK/DOCS/IIT G/Courses/Sem 8/EE 624/GMM Train'
train_files = [f for f in os.listdir(train_directory) if (f.endswith('.wav'))]
test_directory='/media/adr/WORK/DOCS/IIT G/Courses/Sem 8/EE 624/GMM Test'
test_files = [f for f in os.listdir(test_directory) if (f.endswith('.wav'))]
x_train=[np.asarray(()) for i in range(n)]
for fname in train_files:
    i=int(fname.split('_')[1])
    fs,audio=read(train_directory+"/"+fname)
    features=extract_features(audio,fs)
    if x_train[i].size==0:
        x_train[i]=features
    else:
        x_train[i]=np.vstack((x_train[i],features))

#gmm=[GMM(n_components = 16, covariance_type='diag',n_init = 3) for i in range(n)]
#gmm=[GMM(n_components = 16, max_iter=100) for i in range(n)]
#gmm=[GMM(n_components = 16, max_iter=200) for i in range(n)]
#gmm=[GMM(n_components = 16, max_iter=300) for i in range(n)]
gmm=[GMM(n_components = 32, max_iter=100) for i in range(n)]
#gmm=[GMM(n_components = 32, max_iter=200) for i in range(n)]
#gmm=[GMM(n_components = 32, max_iter=300) for i in range(n)]
for i in range(len(x_train)):
    gmm[i].fit(x_train[i])

y_pred=[]
y_true=[]
for fname in test_files:
    val=int(fname.split('_')[1])
    y_true.append(val)
    fs,audio=read(test_directory+"/"+fname)
    features=extract_features(audio,fs)
    log_likelihood=np.zeros(len(gmm))
    for i in range(len(gmm)):
        scores=np.array(gmm[i].score_samples(features))
        log_likelihood[i]=scores.sum()
    winner=np.argmax(log_likelihood)
    y_pred.append(winner)
C=cm(y_true,y_pred,labels=[0,1,2,3,4,5,6,7,8,9])
print(C)
