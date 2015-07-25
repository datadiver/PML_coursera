#PML Project: Weight Lifting Activity Recognition
Juan José Garcés Iniesta
Jul-2015

#1.- Introduction and objective
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of the project is to predict the manner in which they did the exercise. 
The source of data for this project is:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3MCvfqKcP

#2. Data cleaning and EDA
Data loading:   With the corresponding options in read.csv function, I load all the empty cells as NA cells. This make easier to clean the incomplete data/columns

```
sourceTraining <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
sourceTesting <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```

Data cleaning: I will use only the columns without NA values as predictors 
```
colsToTake <- colSums(is.na(sourceTraining))==0
```
I must use the same columns in train and test set. So I'll use the same index vector to clean it.  
```
cleanTrainData <- sourceTraining[ ,colsToTake]
cleanTestData <- sourceTesting[ ,colsToTake]
```

An examination of the data shows that there are several columns with no relevance information for our purpose (1:7)
* Col 1 Index (not relevant)
* Col 2 user_name (not relevant)
* Col 3:7 time and window number (not relevant)
```
cleanTrainData <- cleanTrainData[ ,c(8:60)]
cleanTestData <- cleanTestData[ ,c(8:60)]
```

Now we have 53 columns/variables from 160 initial variables. The last one is different in each set. In training set, is the class (A, B, C, D, E) of this observation. In test set is "problem_id", and the class must be predicted by the model constructed
Now I plot an histogram to see the classe distribution of the full cleaned set of data: 
```
hist(as.numeric(cleanTrainData$classe), 
     main="Distribution of class in train set (19622 obs.)", 
     xlab="Class", col="blue", xaxt="n")
axis(1, 1:5, c("A","B","C","D","E")
```
	

#3. Modelling
The Random Forest algorithm was discussed in the third week of the Practical Machine Learning Course. Due his effectivity and performance, it’s widely used and this is the Model of Choice in this project.
I shall now train a model with 66% of the pmlTraining data set.
```
#Partitioning
> inTrain <- createDataPartition(y=cleanTrainData$classe, p=0.66, list=FALSE)
> training <- cleanTrainData[inTrain, ]
> testing <- cleanTrainData[-inTrain, ]

> modelRF<-train(classe ~., data=training, method="rf")

# Saving model in order to be used later if necessary:
> saveRDS(modelRF, "model_RF.RDS")

> modelRF
Random Forest 

12953 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 12953, 12953, 12953, 12953, 12953, 12953, ... 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
   2    0.9867024  0.9831738  0.001839452  0.002331479
  27    0.9881357  0.9849888  0.001729697  0.002187809
  52    0.9777018  0.9717882  0.004238359  0.005359465

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27. 
```

It’s useful to study the relative importance of each variable. 





The In-Sample Error or Resubstitution Error is defined “as the error rate you get on the same data set you used to build your predictor”.
```
> InTable <- confusionMatrix(Inpredictions, training$classe)
> InTable
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 3683    0    0    0    0
         B    0 2507    0    0    0
         C    0    0 2259    0    0
         D    0    0    0 2123    0
         E    0    0    0    0 2381

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9997, 1)
    No Information Rate : 0.2843     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

We can observe a perfect prediction (Accuracy = 1 = 100% success). This is because I have compared the results of the prediction with the same data set used to train the model.  I have to evaluate the Out-Sample error in order to check there is no overfiting.

#4. Evaluation
Now I’ll test the fifted model with the test data set (testing) and I will check the results in order of Accuracy
```
> Outpredictions <- predict(modelRF, testing)
> OutTable <- confusionMatrix(Outpredictions, testing$classe)
> OutTable
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1891    6    0    0    0
         B    4 1281    6    0    2
         C    2    3 1154   10    1
         D    0    0    3 1082    3
         E    0    0    0    1 1220

Overall Statistics
                                          
               Accuracy : 0.9939          
                 95% CI : (0.9917, 0.9956)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9922          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9968   0.9930   0.9923   0.9899   0.9951
Specificity            0.9987   0.9978   0.9971   0.9989   0.9998
Pos Pred Value         0.9968   0.9907   0.9863   0.9945   0.9992
Neg Pred Value         0.9987   0.9983   0.9984   0.9980   0.9989
Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
Detection Rate         0.2836   0.1921   0.1730   0.1622   0.1829
Detection Prevalence   0.2845   0.1939   0.1754   0.1631   0.1831
Balanced Accuracy      0.9978   0.9954   0.9947   0.9944   0.9975

```
We have a very high accuracy (99,39%) with the model trained. Only 41 wrong predictions from 6669 cases.
Now I’ll use the model fifted to predict the sourceTesting data given (pml-testing.csv) in order to complete the project task.
```
> projectpredictions<-predict(modelRF, cleanTestData)

> projectpredictions
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E

> answers<-as.character(projectpredictions)
> answers
 [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"

```
##Out-of- Sample error
As a project requeriment, a Out-of-Sample error must be evaluated via cross-validation. Out-of-sample error or the generalization error is defined as “ the error you get on a new data set”.   
To make this in a “manual” way, I have splitted  the training data into two groups of 50% (aprox.)of the total training data set: cv01 and cv02.
First, I use cv01 as training set to fit a model “modelcv01”, and I use cv02 data set to validate it and see results. Later, I use cv02 as training set to fit a model “modelcv02”, and I use cv01 data set to validate it and see results. The average accuracy will be more representative from a real behavior of the model with new data.
```
> inTraincv <- createDataPartition(y=training$classe, p=0.5, list=FALSE)
> cv01<-training[inTraincv, ]
> cv02<-training[-inTraincv, ]

> modelcv01<-train(classe~., data=cv01, method="rf")
> modelcv02<-train(classe~., data=cv02, method="rf")

> cv01predict<-predict(modelcv01, cv02)

> cv01Table<-confusionMatrix(cv01predict, cv02$classe)
> cv01Table
Confusion Matrix and Statistics
          Reference
Prediction    A    B    C    D    E
         A 1834   26    0    0    0
         B    5 1219   20    0    2
         C    0    8 1101   25    2
         D    0    0    8 1034    7
         E    2    0    0    2 1179

Overall Statistics
                                          
               Accuracy : 0.9835          
                 95% CI : (0.9801, 0.9864)
    No Information Rate : 0.2844          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9791          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9962   0.9729   0.9752   0.9746   0.9908
Specificity            0.9944   0.9948   0.9935   0.9972   0.9992
Pos Pred Value         0.9860   0.9783   0.9692   0.9857   0.9966
Neg Pred Value         0.9985   0.9935   0.9948   0.9950   0.9979
Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2833   0.1883   0.1701   0.1597   0.1821
Detection Prevalence   0.2873   0.1925   0.1755   0.1620   0.1827
Balanced Accuracy      0.9953   0.9838   0.9843   0.9859   0.9950

> cv02predict<-predict(modelcv02, cv01)

> cv02Table<-confusionMatrix(cv02predict, cv01$classe)
> cv02Table
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1834   26    0    0    0
         B    5 1214    5    1    1
         C    3   13 1108   12    6
         D    0    1   17 1046    4
         E    0    0    0    3 1180

Overall Statistics
                                          
               Accuracy : 0.985           
                 95% CI : (0.9818, 0.9878)
    No Information Rate : 0.2843          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9811          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9957   0.9681   0.9805   0.9849   0.9908
Specificity            0.9944   0.9977   0.9936   0.9959   0.9994
Pos Pred Value         0.9860   0.9902   0.9702   0.9794   0.9975
Neg Pred Value         0.9983   0.9924   0.9959   0.9970   0.9979
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2831   0.1874   0.1710   0.1614   0.1821
Detection Prevalence   0.2871   0.1892   0.1763   0.1648   0.1826
Balanced Accuracy      0.9950   0.9829   0.9871   0.9904   0.9951
```
With “modelcv01” we have 107 errors from 6474  and with “modelcv02” we have 97 errors from 6479. 
The Out-of-sample error is (107+97)/12953 = 0.0157 = 1,57%  

#5. Conclusion
The model fifted using Random forest has an excellent accuracy in the studied problem.  It has been obtained a 99% accuracy with testing data set and more than 98% accuracy in Out-of-Sample error studied with 2-fold cross validation.
