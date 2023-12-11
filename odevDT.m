function []=odevDT(X,Y)
% Decision Tree
%rng('default')

%%
nSample=size(X,1);
%Shuffle Data
S=randperm(nSample);
X=X(S,:); Y=Y(S,:);

bizDT = fitctree(X,Y,'SplitCriterion','gdi','MaxNumSplits', 3,...
    'MinLeafSize',22,'Surrogate', 'off','ClassNames', [0; 1]);
pModel=crossval(bizDT,'KFold',5);
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
title("Decision Tree")
ax = gca;
ax.FontSize = 30;
figure;
end
