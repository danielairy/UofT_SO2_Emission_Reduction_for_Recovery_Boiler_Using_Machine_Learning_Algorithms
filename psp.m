function [f p]=psp(x,fs,r,pl)

% plots power spectrum
% x = data series as vector
% fs= sampling frequency
% r = range to display plot, example : [10 70]
% f = output frequency axis
% p = output power
% pl= number of plots (normal, semilog, loglog)

[nr nc]=size(x);
if nr>nc
    x=x';
end

n=length(x);

if exist('fs')~=1 
    fs=1;
end
if exist('pl')~=1 
    pl=1; 
end
if exist('r')~=1 
    r=[0 0.5]; 
end

if isempty(fs) 
    fs=1; 
end
if isempty(pl) 
    pl=1; 
end
if isempty(r) 
    r=[0 0.5]; 
end


y=fft(x,n);
pp=y.*conj(y)/n;
pp(1)=0;
f = fs*(0:(n/2))/n;
p=pp(1:((n/2)+1));

fl=length(f);
if exist('r')==1
    r=(r/fs)*n;
    mn=r(1,1);mx=r(1,2);
    if (mn>=0|mx<=(n/2))
        f=fs*(mn:mx)/n;
        p=p((mn+1):(mx+1));
    end
end

figure;
plot(f,p)
grid on
title('Frequency Content Normal Plot')
xlabel('Frequency (Hz)')

if pl>=2
    figure;
    semilogx(f,p)
    grid on
    title('Frequency Content Semi Log Plot')
    xlabel('Frequency (Hz)')
end

if pl>=3
    figure;
    loglog(f,p)
    grid on
    title('Frequency Content Log Plot')
    xlabel('Frequency (Hz)')
end
