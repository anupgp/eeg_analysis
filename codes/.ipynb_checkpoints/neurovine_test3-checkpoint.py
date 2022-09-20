import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.integrate import simps
from scipy import stats 
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import sklearn.decomposition

def find_peak_freq_power(psd,freqs,low,high):
    # find the peak frequency and power of a particular frequency band [low>= freq <= high]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_band = freqs[idx_band]
    psd_band = psd[idx_band]
    ipeak_psd = np.where(psd_band>=np.max(psd_band))[0][0]
    peak_freq = freq_band[ipeak_psd]
    peak_psd = psd_band[ipeak_psd]
    # print("low, high: ",low,high)
    # print("freqs: ",freqs)
    # print("select freqs: ",freqs[idx_band])
    # print("PSD: ",psd)
    # print("select PSD: ",np.log10(psd[idx_band])*10, "dB")
    # print("max PSD: ",psd[ipeak_psd])
    # print("max PSD: ",np.log10(peak_psd)*10," dB")
    # input('key')
    return(peak_freq,peak_psd)

def compute_band_power(Sxx,freqs,bands):
    # Sxx: a matrix of power spectrogram per time [nfreqs,ntimebins]
    # bands: a list containing the range for each frequency band
    # fregs: list of frequencies
    abspower = np.zeros((Sxx.shape[1],len(bands)))
    for i in np.arange(0,Sxx.shape[1]):
        for j in np.arange(0,len(bands)):
            low = bands[j][0]
            high = bands[j][1]
            idx_range = np.logical_and(freqs >= low, freqs <= high)
            abspower[i,j] = simps(Sxx[idx_range,i],dx = freqs[1]-freqs[0])
            abspower[i,j] = simps(Sxx[idx_range,i],dx = freqs[1]-freqs[0])
    return(abspower)
                        
        
def compute_1samp_t_test_of_across_time(power):
    # compute t_test across time for each frequency band
    means = np.zeros((power.shape[1]))
    stds = np.zeros((power.shape[1]))
    pvalues = np.zeros((power.shape[1]))
    
    for i in np.arange(0,power.shape[1]):
        means[i] = power[:,i].mean()
        stds[i] = power[:,i].std()
        tvalue,pvalues[i] = stats.ttest_1samp(power[:,i],means[i])
        
    return(pvalues)

def compute_statistics(power):
    # compute t_test across time for each frequency band
    means = np.zeros((power.shape[1]))
    stds = np.zeros((power.shape[1]))
    
    for i in np.arange(0,power.shape[1]):
        means[i] = power[:,i].mean()
        stds[i] = power[:,i].std()
        
    return(means,stds)

def compute_t_test_bw_groups_across_time(power1,power2):
    # compute t_test between groups for each frequency band
    pvalues = np.zeros((power1.shape[1]))
    for i in np.arange(0,power1.shape[1]):
        tvalue,pvalues[i] = stats.ttest_ind(power1[:,i],power2[:,i],equal_var=False)
    return(pvalues)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


datapath = "/Users/macbookair/goofy/data/machine_learning/neurovine/"
aa_name = "AA_EEG_data.csv"
bb_name = "BB_EEG_data.csv"
adata = pd.read_csv(os.path.join(datapath,aa_name))
bdata = pd.read_csv(os.path.join(datapath,bb_name))
channels = ["AF7","FPz","AF8"]
channel = channels[2]
# create time axis
si = 1/200
adata["t"] = np.arange(0,(len(adata))*si,si)
bdata["t"] = np.arange(0,(len(bdata))*si,si)
print(adata)

t0 = 5
fs = 200
lowcut = 0.8
highcut = 40

ay = adata[channel]
by = bdata[channel]
ayf = butter_bandpass_filter(ay,lowcut,highcut,fs)
byf = butter_bandpass_filter(by,lowcut,highcut,fs)
ayf2 = ayf[adata["t"]>t0]
byf2 = byf[bdata["t"]>t0]


freqs,a_psd = signal.welch(ayf2,fs=200,nperseg=200,noverlap=100,nfft=200,detrend=False,return_onesided=True)
freqs,b_psd = signal.welch(byf2,fs=200,nperseg=200,noverlap=100,nfft=200,detrend=False,return_onesided=True)
a_psd = a_psd[(freqs>=1) & (freqs <=30)]
b_psd = b_psd[(freqs>=1) & (freqs <=30)]
freqs = freqs[(freqs>=1) & (freqs <=30)]
psds = np.concatenate((a_psd[:,None],b_psd[:,None]),axis=1)

# compute periodogram (freqs x psd for each time length)
afreqs,at,aSxx = signal.spectrogram(ayf2,fs=200,nperseg=200,noverlap=100,nfft=200,detrend=False,return_onesided=True)
bfreqs,bt,bSxx = signal.spectrogram(byf2,fs=200,nperseg=200,noverlap=100,nfft=200,detrend=False,return_onesided=True)
Sxx = [aSxx,bSxx]

# compute average band power
delta_range = [1,3]
theta_range = [4,6]
alpha_range = [7,12]
beta_range = [13,30]
full_range = [1,30]
all_ranges = [delta_range,theta_range,alpha_range,beta_range,full_range]

# computer powers per time for frequency bands
a_abspower = compute_band_power(aSxx,afreqs,all_ranges)
b_abspower = compute_band_power(bSxx,bfreqs,all_ranges)

# plot color plot of band powers
fh = plt.figure()
ah1 = fh.add_subplot(211)
ah1.plot(at[at<=30],a_abspower[at<=30,:],linewidth=2)
ah2 = fh.add_subplot(212,sharex=ah1)
ah2.plot(bt[bt<=30],b_abspower[bt<=30,:],linewidth=2)
ah1.legend(["delta","theta","alpha","beta","total"],fontsize=12,position="North")
plt.show()

# create dataframe for machine learning
data = np.concatenate((a_abspower[at<=30,:],b_abspower[bt<=30,:]),axis=0)
labels = np.concatenate((np.repeat("AA",len(at[at<=30])),np.repeat("BB",len(bt[bt<=30]))),axis=0)
df = pd.DataFrame({"cat":labels,"delta":data[:,0],"theta":data[:,1],"alpha":data[:,2],"beta":data[:,3],"total":data[:,-1]})
# 
# print(df.head())
# convert categorical string lables to numbers 
le = LabelEncoder()
df['labels'] = le.fit_transform(df['cat'])
# X = df[["delta","theta","alpha","beta","total"]].to_numpy()
# X = df[["delta","alpha"]].to_numpy()
X = df[["theta","alpha"]].to_numpy()
# X = df[["delta","theta","alpha","beta"]].to_numpy()
y = df["labels"].to_numpy()
# data normalization: mean = 0 & std = 1
X = (X - X.mean(axis=0))/X.std(axis=0)
print("Mean: {}, std: {}".format(X.mean(axis=0),X.std(axis=0)))
# ------------------------------
# dimensionality reduction
pca = sklearn.decomposition.PCA().fit(X)
print(pca.transform(X).shape)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
# X = pca.transform(X)[:,:2]      # select the first two components
# -----------------------------
# split data into test and training
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
# fit using SVC from sklearn.svm class
svc = svm.SVC(kernel='linear', C=1).fit(X, y)
# svc = svm.SVC(kernel='logistic', C=1).fit(X, y)
# svc = svm.SVC(kernel='poly', degree=2, C=1)
# svc = svm.SVC(kernel='rbf', gamma=0.7, C=1) # Gaussian kernel
# svc = svm.LinearSVC(C=C)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print("Classification accuracy: {}".format(accuracy_score(y_test,y_pred)))
print("Confusion matrix \n {}".format(confusion_matrix(y_test,y_pred)))
print("Classification report \n {}".format(classification_report(y_test,y_pred)))
input('key')
# scatter plot of data for visualization
# fh = plt.figure()
# ah = fh.add_subplot(111)
# ah.scatter(X[:,0],X[:,1],c=y,cmap="winter")
# plt.show()
# create a mesh to plot in
h = 0.02                        # step size in mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = svc.predict(np.concatenate((xx.ravel()[:,None],yy.ravel()[:,None]),axis=1))
fh = plt.figure()
ah = fh.add_subplot(111)
ah.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.coolwarm, alpha=0.8)
ah.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

