close all
clear variables
file = importfile("05. SO2data.xlsx")
%converting datnum to date string
%Default excel date numbers represent the number of days that have passed since Jan 1, 1900 while matlab data number represent the number of days that have passed since Jan 1, 0000. In terms of date number the difference between excel and matlab is 693960
%timestring = datestr(file.data(:,1) + 693960*ones(1951,1))
file.data(:,1) = file.data(:,1) + 693960*ones(1951,1)

% is there periodic oscillation in SO2 Emission oscillating ?
L = zeros(26,1);
for i = 1:26
    % i is the index of for variables. perform cross corelation with laged
    % copies of Y from -1000 to +1000 with SO2HR (tag20) 
[r,lag] = xcorr(file.data(:,20),file.data(:,i),1000,'coeff'); % corelation is stored in r and lag time is stored in lag
[M,I] = max(r); % find the maximum of corelation in M and report the index in I
L(i) = lag(I); % use index I to find the lag time where cross corelation is maximized and stored in L 
% by the end of the loop L contains all the lag time where corelation is
% maximized
end 

Llog = zeros(26,1);
Mlog = zeros(26,1);
for i = 1:26
[r,lag] = xcorr(log(file.data(:,20)+1),file.data(:,i),1000,'coeff');
[M,I] = max(r);
Llog(i) = lag(I);
Mlog(i)= M;
end 
Llog

Lnorm = zeros(26,1);
Mnorm = zeros(26,1);
for i = 1:26
[r,lag] = xcorr(normalize(file.data(:,20)),normalize(file.data(:,i)),1000,'coeff');
[M,I] = max(r);
Lnorm(i) = lag(I);
Mnorm(i) =M;
end 
Lnorm;

plot(1:26, Mlog,1:26,Mnorm)
xlabel('Tag')
ylabel('Maximum Correlation')
title('Maximum correlation calculated using SO2 vs log SO2 minute average emission')
legend('Log SO2','SO2')

plot(1:26, Llog,1:26,Lnorm)
xlabel('Tag')
ylabel('Lag')
title('Lag time calculated using SO2 vs loged SO2 minute average emission')
legend('Log SO2','SO2')

% Autocorrelation for SO2 Emission
[r,lag] = xcorr(log(file.data(:,20)+1),1000,'coeff');
stem(lag,r)
title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
xlabel('lags')
ylabel('Correlation')
[M,I] = max(r);
lag(I)

% optional plotting 
% is there periodic oscillation in 
% [r,lag] = xcorr(log(file.data(:,20)),file.data(:,7),1000,'coeff');
% stem(lag,r(1:2001))
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
% [r,lag] = xcorr(normalize(file.data(:,20)),normalize(file.data(:,3)),1000,'coeff');
% stem(lag,r(1:2001))
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
% [r,lag] = xcorr(normalize(file.data(:,20)),normalize(file.data(:,3)),1000,'coeff');
% stem(lag,r(1:2001))
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
% [r,lag] = xcorr(file.data(:,20),file.data(:,3),1000,'coeff');
% stem(lag,r(1:2001))
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
% plot(normalize(file.data(:,20)))
% 
% [r,lag] = xcorr(log(file.data(:,20)),file.data(:,3),1000,'coeff');
% stem(lag,r)
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
% [r,lag] = xcorr(normalize(log(file.data(:,20)+1)),normalize(file.data(:,3)),1000,'coeff');
% stem(lag,r)
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)
% 
[r,lag] = xcorr(log(file.data(:,20)+1),file.data(:,2),50,'coeff');
stem(lag,r)
title('Cross-correlation of Primary Air Flow with Log SO2 Minute Average Emission')
xlabel('lags')
ylabel('Correlation')
[M,I] = max(r);
lag(I)
% 
% [r,lag] = xcorr(log(file.data(:,20)+1),file.data(:,7),1000,'coeff');
% stem(lag,r)
% title({'Autocorrelation of SO2 Emission Minute Average','Is Emission Oscillating?'})
% xlabel('lags')
% ylabel('Correlation')
% [M,I] = max(r);
% lag(I)