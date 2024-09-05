def CNN_SVM():
    import glob
    import cv2
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report
    from sklearn.metrics import auc
    from sklearn.model_selection import train_test_split
    from keras.callbacks import CSVLogger


    #read normals files
    normals=[]
    main_path1='D:/A Researcher/Articles/COVID-19/dataset/0 All Converted/X-Ray/Normal/'
    normals=glob.glob(main_path1+'*.jpg')

    #read sicks files
    sicks=[]
    main_path2='D:/A Researcher/Articles/COVID-19/dataset/0 All Converted/X-Ray/Sick/'
    sicks=glob.glob(main_path2+'*.jpg')


    #load normal files
    labels_n=[]
    train_data_n=[]
    for id in normals:    
        img=cv2.imread(id)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  
        img=cv2.resize(img,(200,200))
        img=img/np.max(img)
        img=img.astype('float32')
        train_data_n.append(img)
        labels_n.append(0)


    #load sick files
    labels_s=[]
    train_data_s=[]
    for id in sicks:    
        img=cv2.imread(id)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)  
        img=cv2.resize(img,(200,200))
        img=img/np.max(img)
        img=img.astype('float32')
        train_data_s.append(img)
        labels_s.append(1)

    train_data_n.extend(train_data_s)
    labels_n.extend(labels_s)

    x=np.array(train_data_n)
    y=np.array(labels_n)

    counter=1
    n_epch=100
    lst_loss=[]
    lst_acc=[]
    lst_net_histories=[]
    lst_reports=[]
    lst_AUC=[]
    lst_matrix=[]

    from sklearn.model_selection import KFold
    kfold = KFold(10,shuffle=True,random_state=0)
    for train, test in kfold.split(x,y):

        lg=CSVLogger(f'./results/logger_svm_fold{counter}.log')

        x_train=x[train]
        x_test=x[test]
        train_labels=y[train]
        test_labels=y[test]

        x_train,x_valid,train_labels,valid_labels=train_test_split(x_train,train_labels,test_size=0.2,random_state=0)

        from keras.utils import np_utils
        y_train=np_utils.to_categorical(train_labels)
        y_test=np_utils.to_categorical(test_labels)
        y_valid=np_utils.to_categorical(valid_labels)

        from keras.models import Sequential
        from keras.layers import Dense,Flatten,Conv1D,MaxPool1D,MaxPool2D,Dropout,Activation,BatchNormalization
        from keras.optimizers import Adam
        from keras.losses import hinge
        from keras.regularizers import l2

        model=Sequential()
        model.add(Conv1D(128,3,padding='same',activation='relu',strides=2,input_shape=(200,200)))
        model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(2,activation='linear',activity_regularizer=l2(0.01)))

        model.compile(optimizer=Adam(),loss=hinge,metrics=['accuracy'])
        net_history=model.fit(x_train, y_train, batch_size=32, epochs=n_epch,validation_data=[x_valid,y_valid],callbacks=[lg])
        model.save(f'./CNN+SVM_fold{counter}.h5')

        lst_net_histories.append(net_history)

        test_loss, test_acc=model.evaluate(x_test,y_test)
        lst_loss.append(test_loss)
        lst_acc.append(test_acc)


        predicts=model.predict(x_test)
        predicts=predicts.argmax(axis=1)
        actuals=y_test.argmax(axis=1)

        fpr,tpr,_=roc_curve(actuals,predicts)
        a=auc(fpr,tpr)
        lst_AUC.append(a)

        r=classification_report(actuals,predicts)
        lst_reports.append(r) 

        c=confusion_matrix(actuals,predicts)
        lst_matrix.append(c)

        counter+=1

