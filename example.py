from predict import Predict
#You can obtain predictions by feeding the processed dataset into the following function，and the training data is in numpy format。 
y_predict=Predict.predict(X_test,X_train,y_train)
