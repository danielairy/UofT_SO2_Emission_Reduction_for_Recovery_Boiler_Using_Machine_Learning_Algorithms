close all
file = importfile("05. SO2data.xlsx")
%converting datnum to date string
%Default excel date numbers represent the number of days that have passed since Jan 1, 1900 while matlab data number represent the number of days that have passed since Jan 1, 0000. In terms of date number the difference between excel and matlab is 693960
%timestring = datestr(file.data(:,1) + 693960*ones(1951,1))
file.data(:,1) = file.data(:,1) + 693960*ones(1951,1)
% turnsout time string is a 'matrix' and each character is a column

%creating log variables for S02 Emission Minute Average and SO2 Hourly
figure
subplot(1,2,1)
hist((file.data(:,20)+1))
title('Histogram of SO2 Emission Minute Average')
xlabel('ppm')
ylabel('count')
%saveas(gcf,'Histogram of SO2 Emission Minute Average.png')

subplot(1,2,2)
hist(log(file.data(:,20)+1))
title('Histogram of Log SO2 Emission Minute Average')
xlabel('ppm in natural log scale')
ylabel('count')
saveas(gcf,'Histogram comparison of log SO2 emission.png')
close

% plotting
figure
plot(file.data(:,1),file.data(:,2),'b-')
dateformat = 6;
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Kilo-pound per hour')
title('Time Trend of Primary Air Flow')
saveas(gcf,'Time Trend of Primary Air Flow.png') %gcf means current figure
close

figure
plot(file.data(:,1),file.data(:,3),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Kilo-pound per hour')
title('Time Trend of Secondary Air Flow')
saveas(gcf,'Time Trend of Secondary Air Flow.png')
close

figure
plot(file.data(:,1),file.data(:,4),'b-')
datetick('x',dateformat)
xlabel('Sample date')
ylabel('Kilo-pound per hour')
title('Time Trend of Tertiary Air Flow')
saveas(gcf, 'Time Trend of Tertiary Air Flow.png')
close

figure
plot(file.data(:,1),file.data(:,5),'b-')
datetick('x',dateformat)
xlabel('Sample date')
ylabel('Inches of H2O')
title('Time Trend of Primary Windbox Pressure')
saveas(gcf, 'Time Trend of Primary Windbox Pressure.png')
close

figure
plot(file.data(:,1),file.data(:,6),'b-')
datetick('x',dateformat)
xlabel('Sample date')
ylabel('Inches of H2O')
title('Time Trend of Secondary Windbox Pressure')
saveas(gcf,'Time Trend of Secondary Windbox Pressure.png')
close

figure
plot(file.data(:,1),file.data(:,7),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Inches of H2O')
title('Time Trend of Tertiarary Windbox Pressure')
saveas(gcf,'Time Trend of Tertiarary Windbox Pressure.png')
close

figure
plot(file.data(:,1),file.data(:,8),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('GPM')
title('Time Trend of Burn Rate')
saveas(gcf, 'Time Trend of Burn Rate.png')
close

figure
plot(file.data(:,1),file.data(:,9),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of White Liquor Sulfidity')
saveas(gcf, 'Time Trend of White Liquor Sulfidity.png')
close

figure
plot(file.data(:,1),file.data(:,10),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of Black Liquor Solid 50/50 Test')
saveas(gcf,'Time Trend of Black Liquor Solid 50_50 Test.png')
close

figure
plot(file.data(:,1),file.data(:,11),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of Flue Gas Oxygen')
saveas(gcf,'Time Trend of Flue Gas Oxygen.png')
close

figure
plot(file.data(:,1),file.data(:,12),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('ppm')
title('Time Trend of Total Reduced Sulfur')
saveas(gcf,'Time Trend of Total Reduced Sulfur.png')
close

figure
plot(file.data(:,1),file.data(:,13),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of Dry Solid Density Transimitter A')
saveas(gcf,'Time Trend of Dry Solid Density Transmitter A.png')
close

figure
plot(file.data(:,1),file.data(:,14),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of Dry Solid Density Transmitter B')
saveas(gcf,'Time Trend of Dry Solid Density Transmitter B.png')
close

figure
plot(file.data(:,1),file.data(:,15),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('^{\circ}F')
title('Time Trend of Primary Air Temperature')
saveas(gcf,'Time Trend of Primary Air Temperature.png')
close

figure
plot(file.data(:,1),file.data(:,16),'b-')
datetick('x',dateformat)
xlabel('Samle Date')
ylabel('^{\circ}F')
title('Time Trend of Secondary Air Temperature')
saveas(gcf,'Time Trend of Secondary Air Temperature.png')
close

figure
plot(file.data(:,1),file.data(:,17),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Kilopound per Hour')
title('Time Trend of Steam Flow Rate')
saveas(gcf,'Time Trend of Steam Flow Rate.png')
close

figure
plot(file.data(:,1),file.data(:,18),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Kilopound Per Hour')
title('Time Trend of Black Liquor Solid Flow Rate')
saveas(gcf,'Time Trend of Black Liquor Solid Flow Rate.png')
close

figure
plot(file.data(:,1),file.data(:,19),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
title('Time Trend of Steam to Dry Solid Ratio')
saveas(gcf,'Time Trend of Steam to Dry Solid Ratio.png')
close

figure
plot(file.data(:,1),file.data(:,20),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('ppm')
title('Time Trend of Sulfur Dioxide Concentration Minute Average')
saveas(gcf,'Time Trend of SO2 Concentration Minute Average.png')
close

figure
plot(file.data(:,1),file.data(:,21),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('ppm')
title('Time Trend of Sulfur Dioxide Concentration Hour Average')
saveas(gcf, 'Time Trend of SO2 Concentration Hour Average.png')
close

figure
plot(file.data(:,1),file.data(:,22),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('PSIG')
title('Time Trend of Black Liquor Pressure')
saveas(gcf, 'Time Trend of Black Liquor pressure.png')
close

figure
plot(file.data(:,1),file.data(:,23),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('^{circ}F')
title('Time Trend of Black Liquor Indirect Heater Temperature')
saveas(gcf, 'Time Trend of Black Liquor Indirect Heater Temperature.png')
close

figure
plot(file.data(:,1),file.data(:,24),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('Hz')
title('Time Trend of Salt Cake Rotary Feeder Speed')
saveas(gcf, 'Time Trend of Salt Cake Rotary Feeder Speed.png')
close

figure
plot(file.data(:,1),file.data(:,25),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('%')
title('Time Trend of Lime Kiln Green Liquor Sulfidity')
saveas(gcf,'Time Trend of Lime Kiln Green Liquor Sulfidity.png')
close

figure
plot(file.data(:,1),file.data(:,26),'b-')
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('ppm')
title('Time Trend of Nitrogen Oxides Corelation')
saveas(gcf,'Time Trend of Nitrogen Oxides Correlation.png')
close

figure
subplot(1,2,1)
histogram(file.data(:,8))
xlabel('Bins in GPM')
ylabel('Frequency')
title('Histogram of Burn Rate')
subplot(1,2,2)
histogram(boxcox(file.data(:,8)))
xlabel('Bins in Box-Cox Transformed GPM')
ylabel('Frequency')
title('Histogram of Burn Rate After Box-Cox Transformation')
saveas(gcf,'Comparison of Histogram of Burn Rate Before and After Box-Cox Transformation.png')
close

figure
subplot(1,2,1)
histogram(file.data(:,9))
xlabel('Bins in GPM')
ylabel('Frequency')
title('Histogram of White Liquor Sulfidity')
subplot(1,2,2)
hist(boxcox(file.data(:,9)))
xlabel('Bins in Box-Cox Transformed GPM')
ylabel('Frequency')
title({'Histogram of White Liquor Sulfidity', 'After Box-Cox Transformation'})
saveas(gcf,'Comparison of Histogram of White Liquor Sulfidity before and After Box-Cox Transformation.png')
close

% plotting filtered data
windowSize = 10; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;
for i = 2:26
file.mafiltered_data(:,i) = filter(b,a,file.data(:,i));
end

figure
plot(file.data(:,1),file.data(:,20),'bo')
hold on
plot(file.data(:,1),file.mafiltered_data(:,20),'r-','LineWidth',2)
datetick('x',dateformat)
xlabel('Sample Date')
ylabel('ppm')
legend('original','MA filtered')
title('MA Filtered Time Trend of Sulfur Dioxide Concentration Minute Average')
saveas(gcf,'MA Filtered Time Trend of SO2 Concentration Minute Average.png')
hold off
close
