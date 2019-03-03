	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%交错网格---非均匀介质二维声波方程(一阶压力--速度)、2阶时间差分、2阶空间差分精度
%%加上边界
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clear,clc
tic
%%***********************震源为Ricker子波*********
dtt=0.0001;
tt=-0.06:dtt:0.06;
fm=30;
A=0.01;
wave=A*(1-2*(pi*fm*tt).^2).*exp(-(pi*fm*tt).^2);
plot(wave),title('震源子波--Ricker子波');
%%***********************************************
%% 模型参数设置
dz=5;         % 纵向网格大小，单位m
dx=5;         % 横向网格大小，单位m
dt=0.0001;    % 时间步长，单位s
T=0.5;        % 波动传播时间，单位s
wave(round(T/dt))=0;    % 将子波后面部分补零
% %% 研究区域
% z=-750:dz:750;   x=-1000:dz:1000;
pml=50;          % 吸收层的网格数
plx=pml*dx;      % 上下吸收层的厚度
plz=pml*dz;      % 左右吸收层的厚度
z=-750-plz:dz:750+plz;   
x=-1000-plx:dx:1000+plx;  % 采样区间
n=length(z);     m=length(x);      % 采样点数
z0=round(n/2);   x0=round(m/2);    % 震源位置
Vmax=0;         % 纵波最大速度
 
%%Setting Velocity & Density
zt=-750-plz:dz/2:750+plz; 
xt=-1000-plx:dx/2:1000+plx;   % 速度与密度采样区间
nt=length(zt);     mt=length(xt);       % 速度与密度采样点数目
V=zeros(n,m);       % 介质速度,m/s
d=zeros(nt,mt);     % 介质密度,kg/m^3
 
%%均匀介质模型
for i=1:n
    for k=1:m
        V(i,k)=2.0e3;
    end
end
for i=1:n
    for k=1:m
        d(2*i,2*k)=2.3e3;
    end
end
 
% % %%层状介质模型
% % for i=1:n
% %     for k=1:m
% %         if i < round(n/3)
% %             V(i,k)=2.3e3;
% %         else
% %             V(i,k)=3.0e3;
% %         end
% %     end
% % end
for i=1:n-1
    for k=1:m-1
        d(2*i+1,2*k)=(d(2*i,2*k)+d(2*(i+1),2*k))/2;
        d(2*i,2*k+1)=(d(2*i,2*k)+d(2*i,2*(k+1)))/2;
    end
end
for i=1:n
    for k=1:m
        if V(i,k) > Vmax
            Vmax=V(i,k);
        end
    end
end
%%**********************衰减系数************************
%% ddx、ddz 即，x方向和z方向的衰减系数
R=1e-6;          % 理论反射系数
ddx=zeros(n,m); ddz=zeros(n,m);
 
for i=1:n
    for k=1:m
        %% 区域1
        if i>=1 & i<=pml & k>=1 & k<=pml
            x=pml-k;z=pml-i;
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);
            ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        elseif i>=1 & i<=pml & k>m-pml & k<=m
            x=k-(m-pml);z=pml-i;
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);
            ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        elseif i>n-pml & i<=n & k>=1 & k<=pml
            x=pml-k;z=i-(n-pml);
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);
            ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        elseif i>n-pml & i<=n & k>m-pml & k<=m
            x=k-(m-pml);z=i-(n-pml);
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);
            ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        %% 区域2
        elseif i<=pml & k>pml & k<m-pml+1
            x=0;z=pml-i;
            ddx(i,k)=0;ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        elseif  i>n-pml & i<=n & k>pml & k<=m-pml
            x=0;z=i-(n-pml);
            ddx(i,k)=0;ddz(i,k)=-log(R)*3*Vmax*z^2/(2*plz^2);
        %% 区域3
        elseif i>pml & i<=n-pml & k<=pml
            x=pml-k;z=0;
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);ddz(i,k)=0;
        elseif i>pml & i<=n-pml & k>m-pml & k<=m
            x=k-(m-pml);z=0;
            ddx(i,k)=-log(R)*3*Vmax*x^2/(2*plx^2);ddz(i,k)=0;
        end
    end
end
% figure(1),imagesc(ddz),title('z方向衰减系数');
% figure(2),imagesc(ddx),title('x方向衰减系数');
%%**************************************************
%%**********************波场模拟********************
p0=zeros(n,m);    p1=zeros(n,m);
px0=zeros(n,m);   px1=zeros(n,m);
pz0=zeros(n,m);   pz1=zeros(n,m);
K=zeros(n,m);     
Vx1=zeros(nt,mt); Vx0=zeros(nt,mt);
Vz1=zeros(nt,mt); Vz0=zeros(nt,mt);
 
for t=dt:dt:T
    p0(z0,x0)=dt*V(z0,x0)^2*wave(round(t/dt));
    for i=2:n-1
        for k=2:m-1
            K(i,k)=d(2*i,2*k)*V(i,k)^2;
            Vz1(2*i+1,2*k)=((1-0.5*dt*ddz(i,k))*Vz0(2*i+1,2*k)-dt*(p0(i+1,k)-p0(i,k))/(d(2*i+1,2*k)*dz))/(1+0.5*dt*ddz(i,k));
            Vx1(2*i,2*k+1)=((1-0.5*dt*ddx(i,k))*Vx0(2*i,2*k+1)-dt*(p0(i,k+1)-p0(i,k))/(d(2*i,2*k+1)*dx))/(1+0.5*dt*ddx(i,k));
            
            pz1(i,k)=((1-0.5*dt*ddz(i,k))*pz0(i,k)-K(i,k)*dt*(Vz1(2*i+1,2*k)-Vz1(2*i-1,2*k))/dz)/(1+0.5*dt*ddz(i,k));
            px1(i,k)=((1-0.5*dt*ddx(i,k))*px0(i,k)-K(i,k)*dt*(Vx1(2*i,2*k+1)-Vx1(2*i,2*k-1))/dx)/(1+0.5*dt*ddx(i,k)); 
            
            p1(i,k)=px1(i,k)+pz1(i,k);
        end
    end
    p0=p1;
    pz0=pz1;px0=px1;
    Vz0=Vz1;Vx0=Vx1;
    for i=1:n-2*pml
        for k=1:m-2*pml
            p(i,k)=p1(i+pml,k+pml);
        end
    end
     imagesc(p1),title('声波波场'),pause(0.0000001);
end
figure(1),imagesc(p1);title('声波波场--交错网格，2阶');
figure(2),imagesc(p);title('声波方程--交错网格、加边界');
% figure(2),imagesc(pz1);title('z分量');
% figure(3),imagesc(px1);title('x分量');
%%*************************************************
