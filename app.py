from email.mime import audio
from flask import Flask, render_template, request
import librosa as lb
import numpy as np
import os
from pydub import AudioSegment

from sklearn.linear_model import LogisticRegression                    # Regression classifier
from sklearn.tree import DecisionTreeClassifier                        # Decision Tree classifier
from sklearn import svm                                                # Support Vector Machine
from sklearn.linear_model import SGDClassifier                         # Stochastic Gradient Descent Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Random Forest and Gradient Boosting Classifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix # Metrics to check the performance of models


def wav2mfcc(file_path, max_pad):

    wave, sample_rate = lb.core.load(file_path)
    mfcc = lb.feature.mfcc(y=wave, sr=sample_rate, n_mfcc=12)
    pad_width = max_pad - mfcc.shape[1]

    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:,0:max_pad]

    return mfcc

def ML_PipelineMFCC(clf_name, audio_MFCC):
    X_train = np.load('./X_trainMFCC.npy')
    Y_train = np.load('./Y_train.npy')

    clf = Classifiers[clf_name]
    fit = clf.fit(X_train, Y_train)
    pred = clf.predict(audio_MFCC)
    return pred;
def calcPred(pred):
    if pred == 0:
        return 0;
    return 1
Classifiers = {
    'SVM':svm.SVC(kernel='sigmoid', C=0.1, probability=True),
    'LR':LogisticRegression(random_state=1,C=5,max_iter=200),
    'SGD':SGDClassifier(loss="squared_hinge", penalty="l1", max_iter=2000, shuffle=False),
    'DTC':DecisionTreeClassifier(random_state=10,min_samples_leaf=2,max_features="log2",criterion="entropy"),
    'KNN':KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'),
    'GBC':GradientBoostingClassifier(random_state=10,n_estimators=800,learning_rate=0.2),
}

app = Flask(__name__)
@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html');

@app.route('/', methods=['POST'])
def predict():
    audioFile = request.files['audioFile']
    name, extension = os.path.splitext(audioFile.filename)
    audio_path = "./audio/" + audioFile.filename
    # if(extension == '.wav'):
    #     audio_path = "./audio/" + audioFile.filename
    # elif(extension == '.m4a'):
    #     track = AudioSegment.from_file(audio_path,  format= 'm4a')
    #     audio_path = track.export(audio_path, format='wav')
    res = []
    audioFile.save(audio_path)
    #model lie detection
    audio_mfcc = wav2mfcc(audio_path, 1000)
    audio_mfcc = np.asfarray(audio_mfcc)
    X = audio_mfcc
    nsamples, n = X.shape
    audio_test_mfcc = X.reshape((nsamples, n))
    X_train = np.load('./X_trainMFCC.npy')
    Y_train = np.load('./Y_train.npy')
    y_test = np.load('./y_test.npy')
    pred_GBC = np.load('./MFCC_GBC.npy')
    clf = Classifiers['GBC']
    fit = clf.fit(X_train, Y_train)

    # pred = ML_PipelineMFCC('SGD',audio_test_mfcc.reshape(1, -1))
    # res.append(calcPred(pred))
    # pred = ML_PipelineMFCC('LR',audio_test_mfcc.reshape(1, -1))
    # res.append(calcPred(pred))
    pred = fit.predict(audio_test_mfcc.reshape(1, -1));
    # res.append(calcPred(pred))
    print(fit.predict_proba(audio_test_mfcc.reshape(1, -1)))
    # print(fit.pro)
    # numTrue = 0;
    # numLie = 0;
    # for i in range(3):
    #     if res[i] == 1:
    #         numTrue += 1
    #     else:
    #         numLie += 1 
    # if(numTrue > numLie):
    #     pred == 1;
    # else:
    #     pred == 0;
    pro = fit.predict_proba(audio_test_mfcc.reshape(1, -1));
    # print(pro[0][0], pro[0][1])
    # accurancy = accuracy_score(y_test,pred_GBC) * 100;
    # accuracy = pro * 100;
    # accurancy = round(accurancy,2);
    if pred == 0:
        answer = 'LIE ' + str(round(pro[0][0] * 100,2)) +'%';
    elif pred == 1:
        answer = 'TRUE ' + str(round(pro[0][1]* 100,2)) + '%';
    os.remove(audio_path) 
    return render_template('index.html',prediction = answer)


if __name__ == '__main__':
    app.run(port=3000, debug=True);
