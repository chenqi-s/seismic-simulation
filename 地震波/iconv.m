function  [R,sinput]=iconv(input,wave)
% 对input信号的进行wave的分解，得到 R 和理论输入sinput

%wave=wave(1:floor(end/5));
T=length(wave);
total=length(input);

wav=wave(end:-1:1);
f3=conv(wave,wav);
t=f3(T:end);
TT=toeplitz(t);
b=zeros(T,1);
b(1)=1;
at=TT\b;
R=conv(input,at);
R=R(1:total);
sinput=conv(wave,R);

subplot(4,1,1),plot(input),title('input'),set(gca,'FontSize',16);
subplot(4,1,2),plot(wave),title('wavelet'),set(gca,'FontSize',16);
subplot(4,1,3),plot(R),title('R '),set(gca,'FontSize',16);
subplot(4,1,4),plot(sinput(1:total)),title('theory of input'),xlabel('t');
set(gca,'FontSize',16);
% wnr2 = deconvwnr(sig, input(2:end-300), 1);
