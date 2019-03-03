function [P,rho,x_xl,z_xl,t_xl]=seis_simul1_1(m,n,T)
%苟利国家生死以，岂因祸福避趋之
%对地震波的数值模拟
%wrong!!
%
if(nargin<1)
    m=100;n=100;
    T=1000;
end
%模型参数：
rho=rho_distrb(m,n);
V=v_distrb(m,n);
x_st=0;z_st=0;t_st=0;
x_en=10000;z_en=10000;t_en=3;
K=V.^2.*rho;
widthnum=24;R=0.001;
%点震源信号参数
f0=30;A=5;
x0=m/2;z0=floor(n/3);t0id=10;
%基础参数：压力张量，时空序列，震源信号
P=zeros(m,n,T,'single');
t_xl=linspace(t_st,t_en,T);
x_xl=linspace(x_st,x_en,m);
z_xl=linspace(z_st,z_en,n);
input=input_sig(A,f0,t0id,t_xl);
dx=x_xl(2)-x_xl(1);
dz=z_xl(2)-z_xl(1);
dt=t_xl(2)-t_xl(1);
%
C2=[ 1.6 -0.20 0.025396825396825  -0.001785714285714 ];
C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016];
for i=2:T-1
        P(:,:,i+1)=ndmatsum(P,[i,i-1],[2,-1],2)...
        +K./rho.*dt^2/dx/dx.*...
        ( diff2(P(:,:,i),1)*C2(1)+diff2k(P(:,:,i),1,2)*C2(2)...
        +diff2k(P(:,:,i),1,3)*C2(3)+diff2k(P(:,:,i),1,4)*C2(4) )...
        +K./rho.*dt^2/dz/dz.*...
        ( diff2(P(:,:,i),2)*C2(1)+diff2k(P(:,:,i),2,2)*C2(2)...
        +diff2k(P(:,:,i),2,3)*C2(3)+diff2k(P(:,:,i),2,4)*C2(4) )...
        -dt*dt./(rho.^2).*K.*...
        ( (diff1k(rho,1,1).*diff1k(P(:,:,i),1,1)*C1(1)...
        +diff1k(rho,1,2).*diff1k(P(:,:,i),1,2)*C1(2)...
        +diff1k(rho,1,3).*diff1k(P(:,:,i),1,3)*C1(3)...
        +diff1k(rho,1,4).*diff1k(P(:,:,i),1,4)*C1(4)...
        +diff1k(rho,1,5).*diff1k(P(:,:,i),1,5)*C1(5))/dx/dx...
        +(diff1k(rho,2,1).*diff1k(P(:,:,i),2,1)*C1(1)...
        +diff1k(rho,2,2).*diff1k(P(:,:,i),2,2)*C1(2)...
        +diff1k(rho,2,3).*diff1k(P(:,:,i),2,3)*C1(3)...
        +diff1k(rho,2,4).*diff1k(P(:,:,i),2,4)*C1(4)...
        +diff1k(rho,2,5).*diff1k(P(:,:,i),2,5)*C1(5))/dz/dz...
        );
    P(x0,z0,i+1)=P(x0,z0,i+1)-K(x0,z0)*dt*dt*input(i+1);
    if sum(sum(isnan(P(:,:,i)),2),1)
        break;
    end
    i
end
seis_plot(P(:,:,1:i),x_xl,z_xl,t_xl,T,rho)
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
    colorbar,caxis([minp,maxp]);
    colormap(gray);
    %wait=waitbar(i/100,[num2str(i),'%','  t= ',num2str(t_xl(i))]);
    title('seismic simulation');xlabel('x/m'),ylabel('z/m');
    set(gca,'FontSize',16)
    %[x,y]=find(P(:,:,i)>0);
    hold on
    %plot(x,y,'.'),axis([1,length(x_xl),1,length(z_xl)]);
    hold off,title(num2str(i));
    pause(0.5);
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
%?
%pre=3*vp/2/width*l



function u1=p1(u0,u00)
%边界条件的u1值


function u2=p2(u0,u00)
%边界条件的u2值


function u3=p3(u0,u00)
%边界条件的u3值
function sum=ndmatsum(A,id,x,len)
[M,N]=size(A(:,:,1));
sum=zeros(M,N);
for i=1:len
    sum=sum+x(i).*A(:,:,id(i));
end


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

