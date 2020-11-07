close all
clear variables
rng default % same sequence of random number everytime we re-run the script
file = load( "-mat", "05. SO2data_outliner_filtered.mat"); % load filtered data using DVA tool "filter outlier"
file2 = load("-mat","lag.mat");  % load lag time calculated using cross correlation
file3 = load("-mat","colheaders.mat"); % load column header from data set / DVA tool
lagtime = file2.Llog; % get the lag variable
%convert emission to log scale. column #20 is minute average emission and
%#21 is hourly average emission.
file.filtered_data(:,20) = log(file.filtered_data(:,20)+1); % add 1 to avoid log(0) which is infinite. Each infinite value are unique and would lead setxor function to create more entries than desired.
file.filtered_data(:,21) = log(file.filtered_data(:,21)+1);
%Transform Burn Rate and Sulfidity using boxcox tranformation.
%Use transformation because they are right or left skewed and regression
%methods
%perfers normality
file.filtered_data(:,8) = boxcox(file.filtered_data(:,8));
file.filtered_data(:,9) = boxcox(file.filtered_data(:,9));
% filtering noise using moving average

windowSize = 20; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
for i = 2:26
file.filtered_data(:,i) = filter(b,a,file.filtered_data(:,i));
end

table = array2table(file.filtered_data,'VariableNames', file3.colheaders); % create a table from filtered data
table.Time_Stamp = datetime(table.Time_Stamp,'ConvertFrom','excel');
TT = table2timetable(table); % TT stands for Time Table
% The creation of timetable use column 1 as index. All the variables is
% shifted by 1. In particular, The dependent variable SO2 minute average
% and hourly average locate at index 19 and 20.
% lag adjustement
%TT_lagadjusted = zeros(1951,25);

for i=2:26
TT_lagadjusted(:,i-1) = lag(TT(:,i-1),lagtime(i)); % adjust data with lagtime calculated by maximizing correlation in cross-correlation calculation.
% timetable and TT_lagadjusted have the same row time. When I shifted a
% variable backward, the last row is filled with NaN. When I shift a
% variable n step backward, the last n rows is filled with NaN.
end

% fill missing
maxabslagtime = max(abs(lagtime)) ;% find the maximum absolute lagtime. 
TT_filledmissing = fillmissing(TT_lagadjusted,'movmean',hours([maxabslagtime 0])); % fill missing value using a moving average with window size of 10.
% the last input is window length,which specified as a positive integer scalar, a two-element vector of positive integers, a positive duration scalar, or a two-element vector of positive durations.
%When window is a positive integer scalar, then the window is centered about the current element and contains window-1 neighboring elements. If window is even, then the window is centered about the current and previous elements. If window is a two-element vector of positive integers [b f], then the window contains the current element, b elements backward, and f elements forward.
%I do not want to center my moving average window.
%I want to calculate moving averages using 34 previous neighboring elements
% to calculate moving averages because I want to fill all missing values and
% 'movemean' method is not iterative. Anything less than 34 results in
% missing value in column 23. Alternatively I can use a loop to fill all
% missing with a smaller window.
%check if all missing is filled. Answer is zero if there is no missing
%values
number_of_missing_per_column = sum(isnan(TT_filledmissing{:,:}))
% compare the effectiveness of lag adjustement
correlation_before_lag_adjust = corr(file.filtered_data(:,20),file.filtered_data);
correlation_after_lag_adjust = corr(TT_filledmissing{:,19},TT_filledmissing{:,:});
plot(correlation_after_lag_adjust,correlation_before_lag_adjust(2:end),'ob')
hold;
plot( [ min(correlation_after_lag_adjust) -1 max(correlation_after_lag_adjust) +1], [ min(correlation_after_lag_adjust) -1 max(correlation_after_lag_adjust) +1],'r-' )
xlabel('Correlation after lag adjustement')
ylabel('Correlation before lag adjustement')
title('Effect of lag adjustement')
saveas(gcf,'effect of lag adjustement.png')
close;
%normalize
TT_normalized = normalize(TT_filledmissing);
% sampling 70% for training

s = size(TT_normalized);
ntest = round(s(1)*0.7); % calculate what is the total number of sample in test set.
TT_train = datasample(TT_normalized,ntest,'Replace',false); %randomly select sample from TT_normalized, without replacement
% the reminining is used for testing
TT_test = setxor(TT_normalized,TT_train,'rows');
% check length of training and test set. The length of the first element
% should equal to the length of TT, which is 1951. if there are missing
% values, setxor treats missing value as unique and the size of test set
% and training set will be larger than TT.
test = size(TT_train) + size(TT_test);
%Create individual vectors for SO2 minute average and hourly average as
%dependant variables for regression. Not sure if regress together as a
%matrix affects accuracy.
Y_train = TT_train{:,[19,20]};
Y_minute_train = TT_train{:,19};
Y_hour_train = TT_train{:,20};
%Create X as independant variables
X_train = TT_train{:,[1:18 21:25]};
% Create test set
Y_test = TT_test{:,[19,20]};
Y_minute_test =TT_test{:,19};
Y_hour_test = TT_test{:,20};
X_test =TT_test{:,[1:18 21:25]};
%Perform Regression
% Multivariate simple least square regression AX = Y. A= Y\A
% This is equivalent to mldivide function
% SMLR = Simple Multivariate Linear Regression
SMLR = X_train\Y_train;
SMLR_minute = X_train\Y_minute_train;
SMLR_hour = X_train\Y_hour_train;
diff1 = SMLR_minute - SMLR(:,1);
diff2 = SMLR_hour - SMLR(:,2);
% Turns out regressing both together yield different results from
% regressing individually. Not sure why.

%visualization SMLR
%plot(Y1,X(:,1))
%testing
Y_pred_mldivide =  X_test*SMLR_minute;
% Calculate Sum of square error
SSE_minute_mldivide = sum((Y_pred_mldivide -Y_minute_test).^2);
p = 23;
n = length(Y_minute_test);
% Calculate Residual of square error
RSE_minute = sqrt(1/(n-p-1)*SSE_minute_mldivide)
% Calculate Total square error
TSS_minute = sum( (Y_minute_test -mean(Y_minute_test) ).^2 )
Rq_minute = 1-SSE_minute_mldivide/TSS_minute

%Result Visualiation
plot(Y_minute_test,Y_pred_mldivide,'ob')
hold;
plot([min(Y_minute_test) max(Y_minute_test)],[min(Y_minute_test) max(Y_minute_test)],'b-')
hold;
xlabel('Observed Response')
ylabel('Predicted Response')
title('Prediction vs Observed of SO2 Emission using Naive MLR') 
saveas(gcf,'Prediction vs Observed of SO2 Emission using Naive MLR.png');
save
% Using fitlm
mdl = fitlm(X_train,Y_minute_train);
Y_pred_fitlm = predict(mdl,X_test);

%LASSO Regression with 10 fold cross validation
[b, fitinfo] = lasso(X_train,Y_minute_train, 'CV',10); 
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');
saveas(gcf,'Trace Plot of Coefficients Fit by Lasso.png')
%Display the suggested value of Lambda
fitinfo.Lambda1SE
% Display the lambda with minimal RMSE.
sqrt(fitinfo.LambdaMinMSE)
%Examine the quality of the fit for the suggested value of Lambda
lambdaindex = fitinfo.Index1SE;
RMSE_minute_train_LASSO = sqrt(fitinfo.MSE(lambdaindex))
df = fitinfo.DF(lambdaindex)
% Examine the plot of cross-validated MSE.
lassoPlot(b,fitinfo,'PlotType','CV');
set(gca,'YScale','log');
saveas(gcf,'Cross-Validated MSE of Lasso Fit.png')
% These are the coefficients of the model
z=b(:,lambdaindex);
% Make prediction about test set
% now compute and plot the predicted versus observed response.
yfit=X_test*z+fitinfo.Intercept(lambdaindex);
figure; plot(Y_minute_test,yfit,'o') ;  hold; 
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_test) +1],'r-' )
title('Predicted Vs Actual measurement values for LASSO')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for LASSO.png')
hold;

%LASSO Analysis
RSS_minute_LASSO = sum((Y_minute_test - yfit).^2)
RSE_minute_LASSO = sqrt(1/(n-22-1)*RSS_minute_LASSO)
Rq_minute_LASSO = 1-RSS_minute_LASSO/TSS_minute

%PCA
[u,s,v] = svd(X_train);
% the diagonal (s) finding the tank of the matrix X_train.
diag(s)
% The matrix is full rank.
% The covariance matrix of the latent variable.
diag(diag(s)*diag(s)'/1365)

% Calculate the % variance contribution by each latent variables.
ds = diag(s);
[m,lends]=size(s);
eigvars = (ds.^2)/(m-1) ; % gives the variance explained by each LV
cumtot=cumsum(eigvars); % gives the cummulative sum of eigenvalues
precum = (cumtot*100/cumtot(lends)); 
%Compare output of precum with the table that appears on the top of the
%next page
Score_Matrix = u*s;% This is the score matrix
% The last column is smaller than 0.1, indicating that the rank is 22.
Loading_Matrix = v;

% Finding latent variables from matrix v
latent = v(:,end)/v(end,end)
% The virtual variable is
virtual = X_train * latent
% The mean of the virtual variable is around zero.
mean(virtual)

% using the pca function
[coeff,score,latent,tsquared,explained,mu] = pca(X_train);

% Plot the first two components
figure()
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('X matrix in training set in the first two principle component space')
saveas(gcf,'X matrix in training set in the first two principle component space.png')
close
% make a scree plot
figure()
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('% Variance Explained by Principle Component')
saveas(gcf,'Percentage of variance explained by Principle Component.png')
close
%plot biplot
Variable_label = string(file3.colheaders([2:19 22:26]))
biplot(coeff(:,1:2),'Scores',score(:,1:2),'Varlabels',Variable_label);
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('biplot for X matrix in the first two principle component space')
saveas(gcf,'Biplot of X matrix in training set in the first two principle component space.png')
close

%Principle Component Regression with all 23 components
% transform to regression coefficient
%training 
betaPCR = regress( Y_minute_train - mean(Y_minute_train ), score);
betaPCR = coeff*betaPCR;
betaPCR = [mean(Y_minute_train) - mean(X_train)*betaPCR ; betaPCR];
% Principle Component Regression with 10 components
betaPCR10 = regress( Y_minute_train - mean(Y_minute_train ), score(:,1:10));
betaPCR10 = coeff(:,1:10)*betaPCR10;
betaPCR10 = [mean(Y_minute_train) - mean(X_train)*betaPCR10 ; betaPCR10];
%testing
yfitPCR = [ones(n,1) X_test]*betaPCR;
yfitPCR10 = [ones(n,1) X_test]*betaPCR10;
% Analysis
RSS_PCR = sum((Y_minute_test -yfitPCR).^2);
Rq_minute_PCR = 1- RSS_PCR/TSS_minute;
RSE_minute_PCR = sqrt(1/(n-p-1)*RSS_PCR)

RSS_PCR10 = sum((Y_minute_test -yfitPCR10).^2);
Rq_minute_PCR10 = 1- RSS_PCR10/TSS_minute;
RSE_minute_PCR10 = sqrt(1/(n-10-1)*RSS_PCR10)

%visualization of PCR
figure; plot(Y_minute_test,yfitPCR,'o') ;  hold; 
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_train) +1],'r-' )
title('Predicted Vs Actual measurement values for PCR')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for PCR.png')
hold;close;

%visualization of PCR10
figure; plot(Y_minute_test,yfitPCR10,'o') ;  hold; 
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_train) +1],'r-' )
title('Predicted Vs Actual measurement values for PCR with 10 components')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for PCR with 10 components.png')
hold;close;

% Partial Least Square
[Xloadings,Yloadings,Xscores,Yscores,betaPLS10,PLSPctVar,MSE_PLS,stats] =plsregress(X_train,Y_train,10,'cv',10);
figure
plot(1:10,cumsum(100*PLSPctVar(2,:)),'-bo')
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in Y');
title('Percent Variance Explained in Y vs Number of PLS components')
saveas(gcf,'Percent Variance Explained in Y vs Number of PLS components.png')
close
%regressing with the test set
yfitPLS = [ones(n,1) X_test]*betaPLS10;
residuals_PLS = Y_test - yfitPLS;
figure
stem(residuals_PLS)
xlabel('Observation');
ylabel('Residual');
title('PLS Prediction Residuals')
saveas(gcf,'PLS Prediction Residual.png')
close

figure;hold;  plot(Y_minute_test,yfitPLS(:,1),'o') ;  
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_train) +1],'r-' )
title('Predicted Vs Actual measurement values for PLS')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for PLS.png')
hold;
close

% Calculate Rsquare for PLS
RSS_PLS = sum((Y_minute_test - yfitPLS).^2);
Rq_minute_PLS = 1-RSS_PLS/TSS_minute
RSE_PLS = sqrt(1/(n-10-1)*RSS_PLS)

% Plot weights of components 
plot(1:10,stats.W,'o-');
legend(Variable_label,'Location','NW')
xlabel('Predictor');
ylabel('Weight');
title('Variable weights for the first 10 predictors')
saveas(gcf,'Variable weights for the first 10 predictors for PLS.png')

%Regression Tree
tree = fitrtree(X_train,Y_minute_train,'CrossVal','on','PredictorSelection','curvature');
tree2 = fitrtree(X_train,Y_minute_train,'OptimizeHyperparameters','auto');
tree3 = fitrtree(X_train,Y_minute_train,'MaxNumSplits',30,'CrossVal','on');
yfittree = predict(tree.Trained{1},X_test)
RSS_TREE = sum((Y_minute_test - yfittree).^2);
Rq_minute_TREE = 1-RSS_TREE/TSS_minute
RSE_TREE = sqrt(1/(n-p-1)*RSS_TREE)
kfoldLoss(tree)
view(tree.Trained{1},'Mode','graph');

imp = predictorImportance(tree.Trained{1});
figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
saveas(gcf,'Regression Tree Predictor Importance Estimates.png')

yfittree3 = predict(tree3.Trained{1},X_test)
RSS_TREE3 = sum((Y_minute_test - yfittree3).^2);
Rq_minute_TREE3 = 1-RSS_TREE3/TSS_minute
RSE_TREE3 = sqrt(1/(n-p-1)*RSS_TREE3)
kfoldLoss(tree3)
view(tree3.Trained{1},'Mode','graph')

figure; plot(Y_minute_test,yfittree,'o') ;  hold; 
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_test) +1],'r-' )
title('Predicted Vs Actual measurement values for Regression Tree')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for Regression Tree.png')
hold;


figure; plot(Y_minute_test,yfittree3,'o') ;  hold; 
plot( [ min(Y_minute_test) -1 max(Y_minute_test) +1], [ min(Y_minute_test) -1 max(Y_minute_test) +1],'r-' )
title('Predicted Vs Actual measurement values for Regression Tree with 30 splits')
xlabel('Observed Normalized SO2 Emission')
ylabel('Predicted Normalized SOE Emission')
saveas(gcf,'Predicted vs Observed Values for Regression Tree with 30 splits.png')
hold;