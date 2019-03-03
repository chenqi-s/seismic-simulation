%% seismic simulation

%对地震波的数值模拟
% 1. 9阶空间精度，低频散；
% 2. PML吸收边界；

% 对节点数选取有要求 

%注释部分部分为备选与扩展项

% By 1653282 沈琪

%% the main function

function [P,rho,V,K,input,x_xl,z_xl,t_xl,uz3,uz2,uz1,uy3,uy2,uy1,ux3,ux2,ux1]=seis_simul(m,n,T,choose)

if(nargin<1)
    m=200;n=200;
    T=1200;choose=11;
end
%parpool(2);  % 打开并行池
%hwait=waitbar(0,'prepare...');   %进度条控制
disp(['X num is ',num2str(m),' ; Z num is ',num2str(n),' ; T num is ',num2str(T)]);

% 模型参数：
x_st=0;z_st=0;t_st=0;
x_en=2000;z_en=2000;
t_en=1e-3*T;
widthnum=40;R=0.001;
global M;global N;
M=m+2*widthnum;N=n+widthnum;
V=3e3*ones(M,N);
rho=readstruct(['model/',num2str(choose),'.png'],M,N,1500,3000,-1);
%V=3e3*ones(M,N);
K=V.^2.*rho;

% 点震源信号参数
f0=30;A=5;
x0=M/2;z0=floor(5);t0id=3;
% 基础参数：压力张量，时空序列，震源信号
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
% PML边界矩阵
uz1=zeros(widthnum,N,3);
uz2=uz1;uz3=uz1;
uy1=uz1;uy2=uz1;uy3=uz1;
ux3=zeros(widthnum,M,3);
ux1=ux3;ux2=ux3;
% seis_modelplot(rho(widthnum+1:end-widthnum,1:end-widthnum),...
%     V(widthnum+1:end-widthnum,1:end-widthnum),...
%     K(widthnum+1:end-widthnum,1:end-widthnum),x_xl,z_xl);
disp('model load success...')
disp('processing the data...')
tstart=tic;      %计时器开始
% pml各边界的阻尼矩阵和速度矩阵
Vxz=V(widthnum:-1:1,:);
Vxy=V(M-widthnum+1:end,:);
Vz=V(:,N-widthnum+1:N);
natx=ones(N,1)*[-5:widthnum-6]*dx;naty=ones(M,1)*[-5:widthnum-6]*dz;
Dxz=3*Vxz/2./(widthnum)/dx*log(1/R).*(natx'/(widthnum-6)/dx).^2;
Dxy=3*Vxy/2/(widthnum)/dx*log(1/R).*((natx')/(widthnum-6)/dx).^2;
Dz=3*Vz/2/(widthnum)/dz*log(1/R).*((naty)/(widthnum-6)/dz).^2;
Vz=Vz';Dz=Dz';
% 对每个时间点求下一个点的P
for i=2:T
    %点（面）震源赋值
     P2(x0,z0)=P2(x0,z0)-K(x0,z0)*dt*dt*input(i);
     % P2(x0,z0+1)=P2(x0,z0+1)+K(x0,z0)*dt*dt*input(i);
   % P2([x0-25:x0+25],z0)=P2([x0-25:x0+25],z0)-K([x0-25:x0+25],z0)*dt*dt*input(i);
    
    P3=2*P2-P1...
        +K./rho.*dt^2.* hdif2(P2,1,dx)+K./rho.*dt^2.* hdif2(P2,2,dz)...
        -dt*dt./(rho.^2).*K.*...
    (hdif1(rho,1,dx).*hdif1(P2,1,dx)+hdif1(rho,2,dz).*hdif1(P2,2,dz));
    % PML边界处理（左、右、下） %????下有问题
    [uz3,uz2,uz1,uzsum]=PML(uz3,uz2,uz1,P2(widthnum:-1:1,:),dx,dz,dt,Vxz,Dxz);
    [uy3,uy2,uy1,uysum]=PML(uy3,uy2,uy1,P2(end-widthnum+1:end,:),dx,dz,dt,Vxy,Dxy);
    [ux3,ux2,ux1,uxsum]=PML(ux3,ux2,ux1,P2(:,end-widthnum+1:end)',dz,dx,dt,Vz,Dz);
    P3(widthnum-5:-1:1,:)=uzsum(6:end,:);
    P3(end-widthnum+6:end,:)=uysum(6:end,:);
    for j=5:widthnum
        P3(1-j+widthnum:end+j-widthnum,N-widthnum+j)=uxsum(j,1-j+widthnum:end+j-widthnum);
    end
    
    % waitbar(i/T,hwait,['running: ',num2str(floor(i/T*100)),'%']);
    
    %防止溢出
%     if any(any(abs(P3)>1e50))
%         disp('wrong:the matrix may will have NAN!')   
%         disp(i);
%         break;
%     end
    
%     if i>600
%         break;
%     end
    %save the data
    if mod(i,10)==0
        P(:,:,i/10)=P3;
%         pcolor(P3),shading interp ; 
    end
    %更替时间点
    P1=P2;
    P2=P3;
    
end
%close(hwait);
tend=toc(tstart);   %计时器结束
disp(['running cost time: ',num2str(tend),' s '])
P=P(widthnum+1:end-widthnum,1:end-widthnum,:);    %删去其pml边界

rho=rho(widthnum+1:end-widthnum,1:end-widthnum);
V=V(widthnum+1:end-widthnum,1:end-widthnum);
K=K(widthnum+1:end-widthnum,1:end-widthnum);
disp('creating the image...')
%seis_Pplot(P,x_xl,z_xl)
%disp('saving the data...')
%save('wavedata.mat',single(P))
%disp('creating the video...')
%savevideo('seismic simulation',P,x_xl,z_xl,t_xl,T);
disp('end')
%% end of main


function d2P=hdif2(P,dim,dx)
%1??5步线性组合的10阶精度2阶差分
C1=[ 1.6667 -0.2381 0.0397 -0.0050 0.0003];  %二阶导数的线性系数
d2P=(diff2(P,dim)*C1(1)+diff2k(P,dim,2)*C1(2)+diff2k(P,dim,3)*C1(3)...
        +diff2k(P,dim,4)*C1(4)+diff2k(P,dim,5)*C1(5))/dx/dx;
%  d2P=diff2(P,dim)*C1(1); %并行计算
%  [m,n]=size(d2P);
%  T=zeros(m,n,4);
%  parfor i=2:5
%      T(:,:,i)=diff2k(P,dim,i)*C1(i);
%  end
%  d2P=(d2P+sum(T,3))/dx/dx;

function dP=hdif1(P,dim,dx)
%1??5步线性组合的9阶精度1阶差分
C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016]/2;  %一阶导数的线性系数
dP=(diff1k(P,dim,1)*C1(1)+diff1k(P,dim,2)*C1(2)+diff1k(P,dim,3)*C1(3)...
        +diff1k(P,dim,4)*C1(4)+diff1k(P,dim,5)*C1(5))/dx;
% dP=diff1k(P,dim,1)*C1(1);  %并行计算
% [m,n]=size(dP);
% T=zeros(m,n,4);
% parfor i=2:5
%     T(:,:,i)=diff1k(P,dim,i)*C1(i);
% end
% dP=(dP+sum(T,3))/dx;

function B=diff2(A,dim)
%单跳步二阶差分，1为dx，2为dz
[M,N]=size(A);
if dim==1
    B=[zeros(1,N);diff(A,2,1);zeros(1,N)];
elseif dim==2
    B=[zeros(M,1),diff(A,2,2),zeros(M,1)];
end
    
function B=diff1k(A,dim,k)
% k步跳步一阶差分，1为dx，2为dz
 [M,N]=size(A);
if dim==1
    B=[zeros(k,N);A(2*k+1:end,:)-A(1:end-2*k,:);zeros(k,N)]/(2*k);
elseif dim==2
    B=[zeros(M,k),A(:,2*k+1:end)-A(:,1:end-2*k),zeros(M,k)]/(2*k);
end

function B=diff2k(A,dim,k)
% k步跳步二阶差分，1为dx，2为dz
 [M,N]=size(A);
if dim==1
    B=[zeros(k,N);A(1:end-2*k,:)+A(2*k+1:end,:)-2*A(k+1:end-k,:);zeros(k,N)];
elseif dim==2
    B=[zeros(M,k),A(:,1:end-2*k)+A(:,2*k+1:end)-2*A(:,k+1:end-k),zeros(M,k)];
end

%% end of diff formate

function [Nu3,Nu2,Nu1,usum]=PML(u3,u2,u1,U,dx,dz,dt,C,D)
% PML边界控制
[m,n,~]=size(u3);

Nu1(:,:,3)=u1cd(u1(:,:,3),u1(:,:,2),U,dt,dx,D,C);
Nu2(:,:,3)=u2cd(u2(:,:,3),u2(:,:,2),u2(:,:,1),U,dt,dx,D,C);
Nu3(:,:,3)=u3cd(u3(:,:,3),u3(:,:,2),U,dt,dz,D,C);
Nu1(:,:,[2,1])=u1(:,:,[3,2]);
Nu2(:,:,[2,1])=u2(:,:,[3,2]);
Nu3(:,:,[2,1])=u3(:,:,[3,2]);

usum=reshape(Nu1(:,:,3)+Nu2(:,:,3)+Nu3(:,:,3),[m,n]);

function unew=u1cd(u0,upa,U,dt,dx,D,C)
% 对u1的计算
unew=1./(1+D.*dt).*(2*u0+(D*dt-1).*upa-dt*dt*(D.*D.*u0-C.^2.*hdif2(U,1,1)/dx/dx));

function unew=u2cd(u0,upa,upapa,U,dt,dx,D,C)
% 对u2的计算
unew=( -dt^3*C.^2.*hdif1(D,1,1).*hdif1(U,1,1)/dx/dx-u0.*(-3-6*dt*D+D.^3*dt^3)...
    -upa.*(3+3*D*dt-1.5*D.^2*dt^2)+upapa )./(1+3*dt*D+1.5*D.^2*dt^2);

function unew=u3cd(u0,upa,U,dt,dz,D,C)
% 对u3的计算
unew=2*u0-upa+dt*dt*C.^2.*hdif2(U,2,1)/dz/dz;

%% end of PML condition

function result=input_sig(A,f,t_stid,t_xl)
%点震源信号：RICHER子波
t=t_xl(t_stid:end)-t_xl(t_stid+40);
result=A*exp(-pi*pi*f*f*t.*t).*(1-2*pi*pi*f*f*t.*t);
result=[zeros(1,t_stid-1),result];
 
function rho=rho_distrb(m,n,width)
%地下密度分布场
rho=3*ones(m,n);
% N=n-width;  % 变密度
%  for i=floor(N/2):n
%      %rho(:,i)=i/n+2.5;
%      rho(:,i)=3.5;
%  end
rho=rho*1e3;

function V=v_distrb(m,n)
%地下速度分布场
V=3*ones(m,n);
% for i=n/2:n
%     V(:,i)=4;
% end
V=V*1e3;

%% end of model setting

function seis_modelplot(rho,V,K,z_xl,x_xl)
% 地质模型画图
figure(1)
pcolor(x_xl,z_xl',rho'),view(0,-90),shading interp,c=colorbar;
c.Label.String = 'kg/m3';xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the density model of seismic simulation')
figure(2)
pcolor(x_xl,z_xl',V'),view(0,-90),shading interp,c=colorbar('southoutside');
c.Label.String = 'm/s';xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the velocity model of seismic simulation')
figure(3)
pcolor(x_xl,z_xl',K'),view(0,-90),shading interp,colorbar('southoutside');
xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the K model of seismic simulation')

function seis_Pplot(P,x_xl,z_xl)
%show the seismic in every step time and model
% maxp=max(max(max(P))); % 参数可调
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
    title(num2str(i));
    pause(0.1);
end

function savevideo(name,P,x_xl,z_xl,t_xl,T)
% 保存动画为视频
vidobj=VideoWriter(name);
open(vidobj);
[m,n,T]=size(P);
% P(abs(P)<0.5e3)=0;
maxp=3e4;    % 参数可调
minp=-3e4;
for i=1:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp,view(0,-90);  
    colorbar,caxis([minp,maxp]);axis equal;
    colormap(gray);%jet/gray/HSV
    title('seismic simulation');xlabel('x/m'),ylabel('z/m');
    title(num2str(i));
    pause(0.1);
   frame=getframe;
   %frame.cdata=imresize(frame.cdata,[minp maxp]);
   writeVideo(vidobj,frame);
end
close(vidobj);

%% end of plot
