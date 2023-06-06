import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    fps = list()
    labels = list()
    for root, dirs, files in os.walk("../data/anti-deepfake-data and code/data/authentic"):
        for file in files:
            if "_128" not in root:
                fp = os.path.join(root, file)
                fps.append(fp)
                labels.append(0)

    for root, dirs, files in os.walk("../data/anti-deepfake-data and code/data/fake"):
        for file in files:        
            fp = os.path.join(root, file)
            fps.append(fp)
            labels.append(1)
    n = len(fps)
    # 10% 10% 80%
    X_train, X_test, y_train, y_test = train_test_split(fps, labels, test_size=0.8, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=888)
    info = []
    for i in range(len(X_train)):
        info.append([X_train[i], y_train[i], "train"])
    for i in range(len(X_val)):
        info.append([X_val[i], y_val[i], "val"])
    for i in range(len(X_test)):
        info.append([X_test[i], y_test[i], "test"])
    cols=['FP', 'isFake', 'SET']
    df=pd.DataFrame(info,columns=cols)
    df.to_csv("cycleGAN_10_10_80.csv", index=False)
    
    # 40% 10% 50%
    X_train, X_test, y_train, y_test = train_test_split(fps, labels, test_size=0.5, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=888)
    info = []
    for i in range(len(X_train)):
        info.append([X_train[i], y_train[i], "train"])
    for i in range(len(X_val)):
        info.append([X_val[i], y_val[i], "val"])
    for i in range(len(X_test)):
        info.append([X_test[i], y_test[i], "test"])
    cols=['FP', 'isFake', 'SET']
    df=pd.DataFrame(info,columns=cols)
    df.to_csv("cycleGAN_40_10_50.csv", index=False)

    # 80% 10% 10%
    X_train, X_test, y_train, y_test = train_test_split(fps, labels, test_size=0.1, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=888)
    info = []
    for i in range(len(X_train)):
        info.append([X_train[i], y_train[i], "train"])
    for i in range(len(X_val)):
        info.append([X_val[i], y_val[i], "val"])
    for i in range(len(X_test)):
        info.append([X_test[i], y_test[i], "test"])
    cols=['FP', 'isFake', 'SET']
    df=pd.DataFrame(info,columns=cols)
    df.to_csv("cycleGAN_80_10_10.csv", index=False)

