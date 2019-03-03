%% seismic simulation
%苟利国家生死以，岂因祸福避趋之
%对地震波的数值模拟
% 1.10阶空间精度，低频散；
% 2.起伏地表；
% 3.PML吸收边界；
% 4.；
% runtime :m＝n时，T＝500，time ~= 5x*x-9*x+10, x=m/100,time ~ T;
% 存在计算机内存空间限制，（m,n,T）=(1000,1000,1000)时，占用为9.7G
% 所以需要合理安排时空间关系
% 对节点数选取有要求 
% 并行？？

%% the main function
function [P,rho,V,K,input,x_xl,z_xl,t_xl]=seis_simul5(m,n,T)

if(nargin<1)
    m=300;n=300;
    T=600;
end
%parpool(4);
hwait=waitbar(0,'prepare...');   %进度条控制
disp(['X num is ',num2str(m),' ; Z num is ',num2str(n),' ; T num is ',num2str(T)]);
%模型参数：
x_st=0;z_st=0;t_st=0;
x_en=2000;z_en=2000;
t_en=min([x_en/m*T/2/5.5e3,1e-3*T]);
widthnum=150;R=0.001;
global M;global N;
M=m+2*widthnum;N=n+widthnum;
rho=rho_distrb(M,N);
rho=readstruct('model/8.png',M,N,2500,4500,-1);
V=v_distrb(M,N);
K=V.^2.*rho;
%点震源信号参数
f0=30;A=5;
x0=M/2;z0=floor(7);t0id=3;
%基础参数：压力张量，时空序列，震源信号
P=zeros(M,N,T/10);
P1=zeros(M,N);
P2=P1;P3=P1;
t_xl=linspace(t_st,t_en,T);
x_xl=linspace(x_st,x_en,m);
z_xl=linspace(z_st,z_en,n);
input=input_sig(A,f0,t0id,t_xl);
dx=x_xl(2)-x_xl(1);
dz=z_xl(2)-z_xl(1);
dt=t_xl(2)-t_xl(1);
seis_modelplot(rho(widthnum+1:end-widthnum,1:end-widthnum),...
    V(widthnum+1:end-widthnum,1:end-widthnum),...
    K(widthnum+1:end-widthnum,1:end-widthnum),x_xl,z_xl);
disp('model load success...')
disp('processing the data...')
tstart=tic;      %计时器开始
%C2=[ 1.6667 -0.2381 0.0397 -0.0050 0.0003];  %二阶导数的线性系数
%C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016];  %一阶导数的线性系数
for i=2:T
    P3=2*P2-P1...
        +K./rho.*dt^2.* hdif2(P2,1,dx)+K./rho.*dt^2.* hdif2(P2,2,dz)...
        -dt*dt./(rho.^2).*K.*...
    (hdif1(rho,1,dx).*hdif1(P2,1,dx)+hdif1(rho,2,dz).*hdif1(P2,2,dz));
    
    P3(x0,z0)=P3(x0,z0)-K(x0,z0)*dt*dt*input(i);
    
    %P3=PML(P3,P2,P1,P0,dt,dx,dz,V,widthnum);
    %pcolor(P(:,:,i+1)),shading interp  %
    waitbar(i/T,hwait,['running: ',num2str(floor(i/T*100)),'%']);
    
     %防止溢出
    if any(any(abs(P3)>1e6))
        disp('wrong:the matrix may will have NAN!')   
        return;
    end
    
    %save the data
    if mod(i,10)==0
        P(:,:,i/10)=P3;
    end
    
    %
    %P0=P1;
    P1=P2;
    P2=P3;
end
close(hwait);
tend=toc(tstart);   %计时器结束
disp(['running cost time: ',num2str(tend),' s '])
P=P(widthnum+1:end-widthnum,1:end-widthnum,:);    %删去其pml边界
rho=rho(widthnum+1:end-widthnum,1:end-widthnum);
V=V(widthnum+1:end-widthnum,1:end-widthnum);
K=K(widthnum+1:end-widthnum,1:end-widthnum);
disp('creating the image...')
seis_Pplot(P,x_xl,z_xl)
%disp('saving the data...')
%save('wavedata.mat',single(P))
%disp('creating the video...')
%savevideo('seismic simulation',P,x_xl,z_xl,t_xl,T);
disp('end')
%% end of main


function d2P=hdif2(P,dim,dx)
C1=[ 1.6667 -0.2381 0.0397 -0.0050 0.0003];  %二阶导数的线性系数
% d2P=(diff2(P,dim)*C1(1)+diff2k(P,dim,2)*C1(2)+diff2k(P,dim,3)*C1(3)...
%         +diff2k(P,dim,4)*C1(4)+diff2k(P,dim,5)*C1(5))/dx/dx;
 d2P=diff2(P,dim)*C1(1);
 [m,n]=size(d2P);
 T=zeros(m,n,4);
 parfor i=2:5
     T(:,:,i)=diff2k(P,dim,i)*C1(i);
 end
 d2P=(d2P+sum(T,3))/dx/dx;

function dP=hdif1(P,dim,dx)
C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016]/2;  %一阶导数的线性系数
% dP=(diff1k(P,dim,1)*C1(1)+diff1k(P,dim,2)*C1(2)+diff1k(P,dim,3)*C1(3)...
%         +diff1k(P,dim,4)*C1(4)+diff1k(P,dim,5)*C1(5))/dx;
dP=diff1k(P,dim,1)*C1(1);
[m,n]=size(dP);
T=zeros(m,n,4);
parfor i=2:5
    T(:,:,i)=diff1k(P,dim,i)*C1(i);
end
dP=(dP+sum(T,3))/dx;

function B=diff2(A,dim)
[M,N]=size(A);
if dim==1
    B=[zeros(1,N);diff(A,2,1);zeros(1,N)];
elseif dim==2
    B=[zeros(M,1),diff(A,2,2),zeros(M,1)];
end
    
function B=diff1k(A,dim,k)
 [M,N]=size(A);
if dim==1
    B=[zeros(k,N);A(2*k+1:end,:)-A(1:end-2*k,:);zeros(k,N)]/(2*k);
elseif dim==2
    B=[zeros(M,k),A(:,2*k+1:end)-A(:,1:end-2*k),zeros(M,k)]/(2*k);
end

function B=diff2k(A,dim,k)
[M,N]=size(A);
if dim==1
    B=[zeros(k,N);A(1:end-2*k,:)+A(2*k+1:end,:)-2*A(k+1:end-k,:);zeros(k,N)];
elseif dim==2
    B=[zeros(M,k),A(:,1:end-2*k)+A(:,2*k+1:end)-2*A(:,k+1:end-k),zeros(M,k)];
end
%% end of diff formate

function Pn=PML(P3,P2,P1,P0,dx,dz,dt,C,widthnum)
global M;global N;
Pn=P3;
%??wrong i need 
%  %i=1:widthnum
%  D1=;
%  C1=;
%  pz1=u1cd(P3,P2)
% 
%  %i=M-widthnum+1:M
%  
% 
% for i=N-widthnum+1:N
%     
% end




function unew=u1cd(u0,upa,U,dt,dx,D,C)

unew=1/(1+dt)*(2*u0+(dt-1)*upa+dt*dt*(D.*u0-C.^2.*diff2(U,1)/dx/dx));

function unew=u2cd(u0,upa,upapa,U,dt,dx,D,C)
%??
unew=( dt^3*C.^2.*diff1k(D,1,1)*diff2(U,1)/dx/dx-u0.*(-3-6*dt*D+D.^3*dt^3)...
    -upa.*(3+3*D*dt-1.5*D.^2*dt^2)+upapa )./(1+3*dt*D+1.5*D.^2*dt^2);


function unew=u3cd(u0,upa,U,dt,dz,D,C)

unew=2*u0-upa+dt*dt*C.^2.*diff2(U,2)/dz/dz;

%% end of PML condition



function result=input_sig(A,f,t_stid,t_xl)
%点震源信号：RICHER子波
t=t_xl(t_stid:end)-t_xl(t_stid+100);
result=A*exp(-pi*pi*f*f*t.*t).*(1-2*pi*pi*f*f*t.*t);
result=[zeros(1,t_stid-1),result];
 
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


%% end of model setting




function seis_modelplot(rho,V,K,z_xl,x_xl)
figure(1)
pcolor(x_xl,z_xl',rho'),shading interp,c=colorbar('southoutside');
c.Label.String = 'kg/m3';xlabel('x/m'),ylabel('z/m');
title('the density model of seismic simulation')
figure(2)
pcolor(x_xl,z_xl',V'),shading interp,c=colorbar('southoutside');
c.Label.String = 'm/s';xlabel('x/m'),ylabel('z/m');
title('the velocity model of seismic simulation')
figure(3)
pcolor(x_xl,z_xl',K'),shading interp,colorbar('southoutside');
xlabel('x/m'),ylabel('z/m');
title('the K model of seismic simulation')

function seis_Pplot(P,x_xl,z_xl)
%show the seismic in every step time and model
% maxp=max(max(max(P)));
% minp=min(min(min(P)));
figure(4)
[m,n,T]=size(P);
maxp=5e3;
minp=-5e3;
for i=1:2:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp,view(0,-90);  
    colorbar,caxis([minp,maxp]);axis equal;
    colormap(jet);%jet/gray/HSV
    title('seismic simulation');xlabel('x/m'),ylabel('z/m');
    hold on
    hold off,title(num2str(i));
    pause(0.1);
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


%% end of plot



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


% function sum=ndmatsum(A,B,x,len)
% global M;global N;
% sum=zeros(M,N);
% for i=1:len
%     sum=sum+x(i).*A(:,:,id(i));
% end
    
