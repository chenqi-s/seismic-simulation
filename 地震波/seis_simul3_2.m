%苟利国家生死以，岂因祸福避趋之
%对地震波的数值模拟
% 1.10阶空间精度，低频散；
% 2.起伏地表；
% 3.PML吸收边界；
% 4.；
% wrong!!
% copyright @ 1653282 SQ
% runtime :m＝n时，T＝500，time ~= 5x*x-9*x+10, x=m/100,time ~ T;
% 存在计算机内存空间限制，（m,n,T）=(1000,1000,1000)时，占用为9.7G
% 所以需要合理安排时空间关系
% 对节点数选取有要求 
% 去掉形式主义后应该还能更快的
% 建议用图像线性叠加函数做？
function [P,rho,x_xl,z_xl,t_xl]=seis_simul3_2(m,n,T)

if(nargin<1)
    m=100;n=100;
    T=1000;
end
hwait=waitbar(0,'prepare...');
disp(['X num is ',num2str(m),' Z num is ',num2str(n),' T num is ',num2str(T)]);
%模型参数：
x_st=0;z_st=0;t_st=0;
x_en=1000;z_en=1000;
t_en=min([x_en/m*T/2/5.5e3,1e-3*T]);
widthnum=24;R=0.001;
global M;global N;
M=m+2*widthnum;N=n+widthnum;
rho=rho_distrb(M,N);
%rho=readstruct('model/7.png',M,N,2500,4500,-1);
V=v_distrb(M,N);
K=V.^2.*rho;
%点震源信号参数
f0=30;A=5;
x0=M/2;z0=floor(1);t0id=10;
%基础参数：压力张量，时空序列，震源信号
P=zeros(M,N,T);
t_xl=linspace(t_st,t_en,T);
x_xl=linspace(x_st,x_en,m);
z_xl=linspace(z_st,z_en,n);
input=input_sig(A,f0,t0id,t_xl);
dx=x_xl(2)-x_xl(1);
dz=z_xl(2)-z_xl(1);
dt=t_xl(2)-t_xl(1);
disp('model load success...')
disp('processing the data...')
%D=zn(widthnum,3000,P,R);
tstart=tic;
C2=[ 1.6667 -0.2381 0.0397 -0.0050 0.0003];
C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016];
for i=2:T-1
    P(:,:,i+1)=ndmatsum(P,[i,i-1],[2,-1],2)+...
        dfx(dfx(P(:,:,i),C1)./rho,C1).*K.*dt^2/dx/dx+dfy(dfy(P(:,:,i),C1)./rho,C1).*K.*dt^2/dz/dz;
%         +K./rho.*dt^2/dx/dx.*...
%         ( diff2(P(:,:,i),1)*C2(1)+diff2k(P(:,:,i),1,2)*C2(2)...
%         +diff2k(P(:,:,i),1,3)*C2(3)+diff2k(P(:,:,i),1,4)*C2(4)+diff2k(P(:,:,i),1,5)*C2(5) )...
%         +K./rho.*dt^2/dz/dz.*... %change
%         ( diff2(P(:,:,i),2)*C2(1)+diff2k(P(:,:,i),2,2)*C2(2)...
%         +diff2k(P(:,:,i),2,3)*C2(3)+diff2k(P(:,:,i),2,4)*C2(4)+diff2k(P(:,:,i),2,5)*C2(5) )...
%         -dt*dt./(rho.^2).*K.*...
%     ((diff1k(rho,1,1)*C1(1)+diff1k(rho,1,2)*C1(2)+diff1k(rho,1,3)*C1(3)...
%         +diff1k(rho,1,4)*C1(4)+diff1k(rho,1,5)*C1(5)).*...
%         (diff1k(P(:,:,i),1,1)*C1(1)+diff1k(P(:,:,i),1,2)*C1(2)+diff1k(P(:,:,i),1,3)*C1(3)...
%         +diff1k(P(:,:,i),1,4)*C1(4)+diff1k(P(:,:,i),1,5)*C1(5))/dx/dx...
%         +(diff1k(rho,2,1)*C1(1)+diff1k(rho,2,2)*C1(2)+diff1k(rho,2,3)*C1(3)...
%         +diff1k(rho,2,4)*C1(4) +diff1k(rho,2,5)*C1(5)).*...
%         (diff1k(P(:,:,i),2,1)*C1(1)+diff1k(P(:,:,i),2,2)*C1(2)+diff1k(P(:,:,i),2,3)*C1(3)...
%         +diff1k(P(:,:,i),2,4)*C1(4) +diff1k(P(:,:,i),2,5)*C1(5))/dz/dz);
    
    
    
    %         ( (diff1k(rho,1,1).*diff1k(P(:,:,i),1,1)*C1(1)...
%         +diff1k(rho,1,2).*diff1k(P(:,:,i),1,2)*C1(2)...
%         +diff1k(rho,1,3).*diff1k(P(:,:,i),1,3)*C1(3)...
%         +diff1k(rho,1,4).*diff1k(P(:,:,i),1,4)*C1(4)...
%         +diff1k(rho,1,5).*diff1k(P(:,:,i),1,5)*C1(5))/dx/dx...
%         +(diff1k(rho,2,1).*diff1k(P(:,:,i),2,1)*C1(1)...
%         +diff1k(rho,2,2).*diff1k(P(:,:,i),2,2)*C1(2)...
%         +diff1k(rho,2,3).*diff1k(P(:,:,i),2,3)*C1(3)...
%         +diff1k(rho,2,4).*diff1k(P(:,:,i),2,4)*C1(4)...
%         +diff1k(rho,2,5).*diff1k(P(:,:,i),2,5)*C1(5))/dz/dz...
%         );

%         [zeros(1,N);diff(rho,1)].*[zeros(1,N);diff(P(:,:,i),1)]/dx/dx...
%         +[zeros(M,1),diff(rho,1,2)].*[zeros(M,1),diff(P(:,:,i),1,2)]/dz/dz...

    



    P(x0,z0,i+1)=P(x0,z0,i+1)-K(x0,z0)*dt*dt*input(i+1);
    %pcolor(P(:,:,i+1)),shading interp
    waitbar(i/T,hwait,['running: ',num2str(floor(i/T*100)),'%']);
    if sum(sum(isnan(P(:,:,i)),2),1)
        disp('wrong:the matrix have NAN!')
        return;
    end
end
close(hwait);
tend=toc(tstart);
disp(['running cost time: ',num2str(tend),' s '])
P=P(widthnum+1:end-widthnum,1:end-widthnum,:);
rho=rho(widthnum+1:end-widthnum,1:end-widthnum);
disp('creating the image...')
%seis_plot(P,x_xl,z_xl,t_xl,T,rho)
disp('end')
%disp('saving the data...')
%save('wavedata.mat',P)
%disp('creating the video...')
%savevideo('seismic simulation',P,x_xl,z_xl,t_xl,T)

%%end of main

function D=dfx(A,C1)
D=diff1k(A,1,1)*C1(1)+diff1k(A,1,2)*C1(2)+diff1k(A,1,3)*C1(3)...
      +diff1k(A,1,4)*C1(4)+diff1k(A,1,5)*C1(5);

function D=dfy(A,C1)
D=diff1k(A,2,1)*C1(1)+diff1k(A,2,2)*C1(2)+diff1k(A,2,3)*C1(3)...
      +diff1k(A,2,4)*C1(4)+diff1k(A,2,5)*C1(5);

function bound_cdt()
%边界条件PML







function result=input_sig(A,f,t_stid,t_xl)
%点震源信号：RICHER子波
t=t_xl(t_stid:end)-t_xl(t_stid);
result=A*exp(-pi*pi*f*f*t.*t).*(1-2*pi*pi*f*f*t.*t);
result=[zeros(1,t_stid),result];


function seis_plot(P,x_xl,z_xl,t_xl,T,rho)
%show the seismic in every step time and model
figure(1)
pcolor(x_xl,z_xl',rho'),shading interp,c=colorbar('southoutside');
c.Label.String = 'kg/m3';xlabel('x/m'),ylabel('z/m');
title('the density model of seismic simulation')
figure(2)
% maxp=max(max(max(P)));
% minp=min(min(min(P)));
[m,n,T]=size(P);
sp=sort(reshape(P(:,:,floor(end*0.2):end),1,m*n*(T-floor(T*0.2)+1)));
maxp=sp(floor(end*0.999));
minp=sp(floor(end*0.002));
% sp=sort(reshape(P,1,m*n*T));
% maxp=sp(floor(end*0.95));
% minp=sp(floor(end*0.05));
for i=1:10:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp;  %wrong 
    colorbar,caxis([minp,maxp]);axis equal;
    colormap();%jet/gray/HSV
    %wait=waitbar(i/100,[num2str(i),'%','  t= ',num2str(t_xl(i))]);
    title('seismic simulation');xlabel('x/m'),ylabel('z/m');
    %[x,y]=find(P(:,:,i)>0);
    hold on
    %plot(x,y,'.'),axis([1,length(x_xl),1,length(z_xl)]);
    hold off,title(num2str(i));
    pause(0.1);
    %close(wait);
end

function savevideo(name,P,x_xl,z_xl,t_xl,T)
vidobj=VideoWriter(name);
open(vidobj);
for i=1:T
   %plot
   frame=getframe;
   %frame.cdata=imresize(frame.cdata,[540 540]);
   writeVideo(vidobj,frame);
end
close(vidobj);


function rho=rho_distrb(m,n)
%地下密度分布场
rho=3*ones(m,n);
%  for i=n/2:n
%      rho(:,i)=i/n+2.5;
%  end
rho=rho*1e3;

function V=v_distrb(m,n)
V=3*ones(m,n);
% for i=n/2:n
%     V(:,i)=4;
% end
V=V*1e3;


function pre=zn(width,vp,P,R)
%阻尼矩阵
[m,n]=size(P(:,:,1));
pre=zeros(m,n);
natx=ones(n,1)*[1:width];naty=ones(m,1)*[1:width];
pre(1:width,:)=3*vp/2/width*log(1/R)*(natx'/width).^2;
pre(end:-1:end-width,:)=3*vp/2/width*log(1/R)*(natx'/width).^2;
pre(:,end:-1:end-width)=3*vp/2/width*log(1/R)*(naty/width).^2;
for i=1:width
    for j=n:-1:n-i
        pre(i,j)=3*vp/2/width*log(1/R)*(i/width).^2;
        pre(end-i,j)=3*vp/2/width*log(1/R)*(i/width).^2;
    end
end


function u1=cd1(u10,u100,u0,u00,dt,v,m,n,D)
%边界条件的u1值
u1=(dt^2*v^2*[zeros(1,n);diff(u0,2,1);zeros(1,n)]-dt^2*D.*u00+2*u10+(dt*D-1).*u100)./(1+dt*D);

function u2=cd2(u0,u00)
%边界条件的u2值


function u3=cd3(u30,u300,u0,dt,v,m,n)
%边界条件的u3值
u3=2*u30-u300+dt^2*v*v*[zeros(m,1),diff(u0,1,2)];


% function pre=sj(P,width,xs)
% % 海绵吸收边界,效果极差（对参数和边界厚度具有较高的契合要求）
% [m,n]=size(P(:,:));
% pre=P;
% natx=ones(n,1)*[width-1:-1:0];naty=ones(m,1)*[width-1:-1:0];
% pre(1:width,:)=pre(1:width,:).*exp(-natx'.^2.5*xs);
% pre(end:-1:end-width+1,:)=pre(end:-1:end-width+1,:).*exp(-natx'.^2.5*xs);
% pre(:,end:-1:end-width+1)=pre(:,end:-1:end-width+1).*exp(-naty.^2.5*xs);

function sum=ndmatsum(A,id,x,len)
global M;global N;
sum=zeros(M,N);
for i=1:len
    sum=sum+x(i).*A(:,:,id(i));
end
    
function B=diff2(A,dim)
global M;global N;
if dim==1
    B=[zeros(1,N);diff(A,2,1);zeros(1,N)];
elseif dim==2
    B=[zeros(M,1),diff(A,2,2),zeros(M,1)];
end
    
function B=diff1k(A,dim,k)
global M;global N;
if dim==1
    B=[zeros(k,N);A(2*k+1:end,:)-A(1:end-2*k,:);zeros(k,N)]/(2*k);
elseif dim==2
    B=[zeros(M,k),A(:,2*k+1:end)-A(:,1:end-2*k),zeros(M,k)]/(2*k);
end

function B=diff2k(A,dim,k)
global M;global N;
if dim==1
    B=[zeros(k,N);A(1:end-2*k,:)+A(2*k+1:end,:)-2*A(k+1:end-k,:);zeros(k,N)];
elseif dim==2
    B=[zeros(M,k),A(:,1:end-2*k)+A(:,2*k+1:end)-2*A(:,k+1:end-k),zeros(M,k)];
end