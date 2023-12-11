function []=odevKNN(X,Y)
% KNN with Cosine metric 
%rng('default')

%%
nSample=size(X,1);
%Shuffle Data
S=randperm(nSample);
X=X(S,:); Y=Y(S,:);

bizKNN = fitcknn(X,Y,'Distance','Cosine','Exponent',[],'NumNeighbors',120,...
        'DistanceWeight', 'Inverse','Standardize', true,'ClassNames', [0; 1]);
pModel=crossval(bizKNN,'KFold',5);
[vPred, ~] = kfoldPredict(pModel);
vAcc = 1 - kfoldLoss(pModel, 'LossFun', 'ClassifError');

[cm, ~] = confusionmat(categorical(Y),categorical(vPred));
TN=cm(1,1); FP=cm(1,2); FN=cm(2,1); TP=cm(2,2);

Accuracy        =   ((TN+TP)/(TN+FP+FN+TP))*100;
Sensitivity     =   (TP/(TP+FN))*100;
Specificity     =   (TN/(TN+FP))*100;
MCC             =   ((TP*TN)-(FN*FP))/ sqrt((TP+FN)*(TN+FP)*(TP+FP)*(TN+FN));
Recall          =   (TP/(TP+FN))*100;
Precision       =   (TP/(TP+FP))*100;
F1_Score        =   (Precision*Recall) / (Precision+Recall);


disp([' Accuracy:    ', num2str(mean(Accuracy)) ]);
disp([' Sensitivity: ', num2str(mean(Sensitivity))  ]);
disp([' Specificity: ', num2str(mean(Specificity))  ]);
disp([' MCC:         ', num2str(mean(MCC))	]);
disp([' F1_Score:    ', num2str(mean(F1_Score))    ]);

%%
confusionchart(categorical(Y),categorical(vPred));
title("KNN")
ax = gca;
ax.FontSize = 30;
figure;
end