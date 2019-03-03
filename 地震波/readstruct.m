function S=readstruct(filename,m,n,vmin,vmax,isadd)
% 读取地质模型，filename为读取的图片文件名称，［m，n］为读取的地地质体的节点数
% ［vmin，vmax］为地质体参数的上下界限， isadd 一般取 －1
imgray=(rgb2gray(imread(filename)))';
[M,N]=size(imgray);
x=floor(linspace(1,M,m));
y=floor(linspace(1,N,n));
S=imgray(x,y);
%imshow(S);
S=double(S)*isadd;
smax=max(max(S));
smin=min(min(S));
S=(vmax-vmin)/(smax-smin)*S+(vmax*smin-vmin*smax)/(smin-smax);


