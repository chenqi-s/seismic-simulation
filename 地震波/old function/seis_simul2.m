%苟利国家生死以，岂因祸福避趋之
%对地震波的数值模拟
% 1.高精度，低频散；
% 2.起伏地表；
% 3.PML吸收边界；
% 4.；
% 5.；
% copyright @ 1653282 SQ
function [P,rho,x_xl,z_xl,t_xl]=seis_simul2(m,n,T)

if(nargin<1)
    m=100;n=100;
    T=1000;
end
%模型参数：
x_st=0;z_st=0;t_st=0;
x_en=10000;z_en=10000;t_en=5;
widthnum=100;R=0.001;
M=m+2*widthnum;N=n+widthnum;
rho=rho_distrb(M,N);
V=v_distrb(M,N);
K=V.^2.*rho;
%点震源信号参数
f0=30;A=5;
x0=M/2;z0=floor(n/2);t0id=10;
%基础参数：压力张量，时空序列，震源信号
P=zeros(M,N,T);
t_xl=linspace(t_st,t_en,T);
x_xl=linspace(x_st,x_en,m);
z_xl=linspace(z_st,z_en,n);
input=input_sig(A,f0,t0id,t_xl);
dx=x_xl(2)-x_xl(1);
dz=z_xl(2)-z_xl(1);
dt=t_xl(2)-t_xl(1);
%D=zn(widthnum,3000,P,R);
%
for i=2:T-1
    P(:,:,i+1)=2*P(:,:,i)-P(:,:,i-1)+K./rho.*[zeros(1,N);diff(P(:,:,i),2,1);zeros(1,N)]...
        *dt^2/dx/dx+K./rho.*[zeros(M,1),diff(P(:,:,i),2,2),zeros(M,1)]*dt^2/dz/dz...
        -dt*dt./(rho.^2).*K.*([zeros(1,N);diff(rho,1)].*[zeros(1,N);diff(P(:,:,i),1)]/dx/dx...
        +[zeros(M,1),diff(rho,1,2)].*[zeros(M,1),diff(P(:,:,i),1,2)]/dz/dz);
    P(x0,z0,i+1)=P(x0,z0,i+1)-K(x0,z0)*dt*dt*input(i+1);
    if sum(sum(isnan(P(:,:,i+1)),2),1)
        disp('wrong:NAN!');
        break;
    end
end
P=P(widthnum+1:end-widthnum,1:end-widthnum,:);
rho=rho(widthnum+1:end-widthnum,1:end-widthnum);
seis_plot(P,x_xl,z_xl,t_xl,T,rho)
%savevideo('seismic simulation',P,x_xl,z_xl,t_xl,T)

%%end of main



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
[m,n,T]=size(P);
% maxp=max(max(max(P)));
% minp=min(min(min(P)));
sp=sort(reshape(P,1,m*n*T));
maxp=sp(floor(end*0.95));
minp=sp(floor(end*0.05));
for i=1:10:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp;  %wrong 
    colorbar,caxis([minp,maxp]);axis equal;
    colormap(jet);
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
for i=n/2:n
    rho(:,i)=3.5;
end
rho=rho*1e3;

function V=v_distrb(m,n)
V=3*ones(m,n);
for i=n/2:n
    V(:,i)=4;
end
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

function ndmatsum(A,x,m,n,len)
sum=zeros(m,n);
for i=1:len
    sum=sum+x(i)*A(:,:,i);
end
    
    
    
    