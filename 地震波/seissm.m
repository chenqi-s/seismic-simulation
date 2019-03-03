function varargout = seissm(varargin)
% SEISSM MATLAB code for seissm.fig
%      SEISSM, by itself, creates a new SEISSM or raises the existing
%      singleton*.
%
%      H = SEISSM returns the handle to a new SEISSM or the handle to
%      the existing singleton*.
%
%      SEISSM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SEISSM.M with the given input arguments.
%
%      SEISSM('Property','Value',...) creates a new SEISSM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before seissm_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to seissm_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help seissm

% Last Modified by GUIDE v2.5 05-Jun-2018 14:39:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @seissm_OpeningFcn, ...
                   'gui_OutputFcn',  @seissm_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before seissm is made visible.
function seissm_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to seissm (see VARARGIN)

% Choose default command line output for seissm
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes seissm wait for user response (see UIRESUME)
% uiwait(handles.figure1);
set(handles.axes1,'visible','off')

pic=imread('model/default.jpg');
axes(handles.axes1);  
imshow(pic);  
vimg=imread(['model/1.png']);
axes(handles.axes5);
imshow(vimg);
A=5;f=30;width=3e3;depth=3e3;
x0id=150;
z0id=150;
t0id=10;
datas=cell(7,2);
datas{1,2}=f;datas{2,2}=A;datas{3,2}=width;datas{4,2}=depth;
datas{5,2}=x0id;datas{6,2}=z0id;datas{7,2}=t0id;
datas{1,1}='fre';datas{2,1}='Amp';datas{3,1}='width';datas{4,1}='depth';
datas{5,1}='expl Xid';datas{6,1}='expl Zid';datas{7,1}='expl Tid';
set(handles.uitable2,'data',datas);
global m;global n;global t;global P;global x_xl;global z_xl;global t_xl;
global mapid;
mapid=1;

% --- Outputs from this function are returned to the command line.
function varargout = seissm_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton1,'enable','off');
global m;global n;global t;global P;global x_xl;global z_xl;global t_xl;
global mapid;global x0id;global z0id;global t0id;global A;
global f0;global width;global depth;global issucc;
mapid=get(handles.popupmenu3,'Value');
m=str2double(get(handles.edit1,'string'));
n=str2double(get(handles.edit3,'string'));
t=str2double(get(handles.edit4,'string'));
datas=get(handles.uitable2,'data');
A=cell2mat(datas(2,2));
f0=cell2mat(datas(1,2));
width=cell2mat(datas(3,2));
depth=cell2mat(datas(4,2));
width=0.5*(width+depth);depth=width;
x0id=cell2mat(datas(5,2));
z0id=cell2mat(datas(6,2));
t0id=cell2mat(datas(7,2));
vch=get(handles.listbox1,'value');
rhoch=get(handles.listbox2,'value');

if issucc==1
minp=-5.5e3;
maxp=5e3;
axes(handles.axes1);
set(handles.axes1,'visible','on')
set(handles.text3,'string',char(get(handles.text3,'string'),'creating the image...'));
for i=1:2:t/5
    pcolor(handles.axes1,x_xl,z_xl',(P(:,:,i)')),shading interp;  %wrong 
    colorbar,caxis([minp,maxp]);axis equal,view(0,-90);
    switch mapid
        case 1
            colormap(hsv); %gray /jet / HSV
        case 2
            colormap(jet);
        case 3
            colormap(gray);
    end
    xlabel('x/m'),ylabel('z/m'),set(gca,'FontSize',13);
    title(['T= ',num2str(i*10),' ms ']);
    set(handles.slider2,'value',floor(i/t*100)/100*5);
    pause(0.1);
end
set(handles.slider2,'value',1);
set(handles.text3,'string',char(get(handles.text3,'string'),'end of simulation'));
end
set(handles.pushbutton1,'enable','on');

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiwait(gcf);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume(gcf);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% global P;
% choice=questdlg('Do you want to save the data?','seismic_simulation','no','yes','no');
% switch choice
%     case 'yes'
%         set(handles.text3,'string',char(get(handles.text3,'string'),'saving the data...'));
%         save sm_data.mat P;
%     case 'no'
%         set(handles.text3,'string',char(get(handles.text3,'string'),'not save the data'));
% end
% set(handles.text3,'string',char(get(handles.text3,'string'),'have save the data...'));
set(handles.pushbutton4,'enable','off');
global m;global n;global t;global P;global x_xl;global z_xl;global t_xl;
global mapid;global x0id;global z0id;global t0id;global A;
global f0;global width;global depth;global issucc;
mapid=get(handles.popupmenu3,'Value');
m=str2double(get(handles.edit1,'string'));
n=str2double(get(handles.edit3,'string'));
t=str2double(get(handles.edit4,'string'));
datas=get(handles.uitable2,'data');
A=cell2mat(datas(2,2));
f0=cell2mat(datas(1,2));
width=cell2mat(datas(3,2));
depth=cell2mat(datas(4,2));
width=0.5*(width+depth);depth=width;
x0id=cell2mat(datas(5,2));
z0id=cell2mat(datas(6,2));
t0id=cell2mat(datas(7,2));
vch=get(handles.listbox1,'value');
rhoch=get(handles.listbox2,'value');
[issucc,P,~,~,~,~,x_xl,z_xl,t_xl]=seis_simul3_1(m,n,t,width,depth,A,f0,x0id,z0id,t0id,vch,rhoch,handles);
set(handles.pushbutton4,'enable','on');

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close 


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1
vchoose=get(handles.listbox1,'value');
global V;
vimg=imread(['model/',num2str(vchoose),'.png']);
axes(handles.axes5);
imshow(vimg);

% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2
rhochoose=get(handles.listbox2,'value');
global rho;
rhoimg=imread(['model/',num2str(rhochoose),'.png']);
axes(handles.axes5);
imshow(rhoimg);

% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global m;global n;global t;global mapid;
set(handles.text3,'string',char(get(handles.text3,'string'),'％％％％％％％％％％％％％％％％％'));
set(handles.text3,'string',char(get(handles.text3,'string'),'model load success'))
m=str2double(get(handles.edit1,'string'));
n=str2double(get(handles.edit3,'string'));
t=str2double(get(handles.edit4,'string'));
set(handles.text3,'string',char(get(handles.text3,'string'),['Xnum is ',num2str(m)]));
set(handles.text3,'string',char(get(handles.text3,'string'),['Znum is ',num2str(n)]));
set(handles.text3,'string',char(get(handles.text3,'string'),['Tnum is ',num2str(t)]));
global V;global rho;
m=str2double(get(handles.edit1,'string'));
n=str2double(get(handles.edit3,'string'));
t=str2double(get(handles.edit4,'string'));
datas=get(handles.uitable2,'data');

width=cell2mat(datas(3,2));
depth=cell2mat(datas(4,2));
width=0.5*(width+depth);depth=width;
x_xl=linspace(0,width,m);
z_xl=linspace(0,depth,n);
vchoose=get(handles.listbox1,'value');
rhochoose=get(handles.listbox2,'value');
V=readstruct(['model/',num2str(vchoose),'.png'],m,n,2500,4500,-1);
rho=readstruct(['model/',num2str(rhochoose),'.png'],m,n,2500,4500,-1);
[X,Z]=meshgrid(x_xl,z_xl);
mesh(handles.axes5,X,Z,(V.^2.*rho)'),shading interp,view(0,-90);
colorbar;
    switch mapid
        case 1
            colormap(hsv); %gray /jet / HSV
        case 2
            colormap(jet);
        case 3
            colormap(gray);
    end
title('K model of struct'),xlabel('x/m'),ylabel('z/m'),zlabel('K');


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
datas=get(handles.uitable2,'data');
m=str2double(get(handles.edit1,'string'));
datas{5,2}=floor(m/2);
set(handles.uitable2,'data',datas);

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function pushbutton1_CreateFcn(hObject, eventdata, handles)

function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes on selection change in listbox3.
function listbox3_Callback(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox3


% --- Executes during object creation, after setting all properties.
function listbox3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function [issucc,P,rho,V,K,input,x_xl,z_xl,t_xl]=seis_simul3_1(m,n,T,x_en,z_en,A,f0,x0,z0,t0id,vch,rhoch,handles)

hwait=waitbar(0,'prepare...');   %进度条控制
%模型参数：
x_st=0;z_st=0;t_st=0;
% x_en=1000;z_en=1000;
t_en=T/1e3;
widthnum=40;R=0.001;
global M;global N;
M=m+2*widthnum;N=n+widthnum;

rho=readstruct(['model/',num2str(rhoch),'.png'],M,N,2500,4500,-1);
V=readstruct(['model/',num2str(vch),'.png'],M,N,2500,4500,-1);

K=V.^2.*rho;
x0=x0+widthnum;
%点震源信号参数
% f0=30;A=5;
%x0=M/2;z0=floor(100);t0id=10;
%基础参数：压力张量，时空序列，震源信号
P=zeros(M,N,T/5);
P1=zeros(M,N);
P2=P1;P3=P1;
t_xl=linspace(t_st,t_en,T);
x_xl=linspace(x_st,x_en,m);
z_xl=linspace(z_st,z_en,n);
input=input_sig(A,f0,t0id,t_xl);
dx=x_xl(2)-x_xl(1);
dz=z_xl(2)-z_xl(1);
dt=t_xl(2)-t_xl(1);
uz1=zeros(widthnum,N,3);
uz2=uz1;uz3=uz1;
uy1=uz1;uy2=uz1;uy3=uz1;
ux3=zeros(widthnum,M,3);
ux1=ux3;ux2=ux3;
set(handles.text3,'string','%%%%%%%%%%%');
% seis_modelplot(rho(widthnum+1:end-widthnum,1:end-widthnum),...
%     V(widthnum+1:end-widthnum,1:end-widthnum),...
%     K(widthnum+1:end-widthnum,1:end-widthnum),x_xl,z_xl);
set(handles.text3,'string',char(get(handles.text3,'string'),'processing the data...'));
%D=zn(widthnum,3000,P,R);
close(hwait);
% hwait=waitbar(0,'Please waiting...','CreateCancelBtn',...
%             'setappdata(gcbf,''canceling'',1)');
hwait=waitbar(0,'Please waiting...');
set(handles.text3,'string',char(get(handles.text3,'string'),'running: 0 %'));
tstart=tic;      %计时器开始
Vxz=V(widthnum:-1:1,:);
Vxy=V(M-widthnum+1:end,:);
Vz=V(:,N-widthnum+1:N);
natx=ones(N,1)*[-1:widthnum-2]*dx;naty=ones(M,1)*[-1:widthnum-2]*dz;
Dxz=3*Vxz/2./widthnum/dx*log(1/R).*(natx'/widthnum/dx).^2;
Dxy=3*Vxy/2/widthnum/dx*log(1/R).*(natx'/widthnum/dx).^2;
Dz=3*Vz/2/widthnum/dz*log(1/R).*(naty/widthnum/dz).^2;
Vz=Vz';Dz=Dz';
for i=2:T
    bftext=get(handles.text3,'string');
    set(handles.text3,'string',char(bftext(1:end-1,:),['running : ',num2str(i/T*100),' %']));
    P2(x0,z0)=P2(x0,z0)-K(x0,z0)*dt*dt*input(i);
   % P2([x0-25:x0+25],z0)=P2([x0-25:x0+25],z0)-K([x0-25:x0+25],z0)*dt*dt*input(i);
    
    P3=2*P2-P1...
        +K./rho.*dt^2.* hdif2(P2,1,dx)+K./rho.*dt^2.* hdif2(P2,2,dz)...
        -dt*dt./(rho.^2).*K.*...
    (hdif1(rho,1,dx).*hdif1(P2,1,dx)+hdif1(rho,2,dz).*hdif1(P2,2,dz));
    % PML边界处理（左、右、下）
    [uz3,uz2,uz1,uzsum]=PML(uz3,uz2,uz1,P2(widthnum:-1:1,:),dx,dz,dt,Vxz,Dxz);
    [uy3,uy2,uy1,uysum]=PML(uy3,uy2,uy1,P2(end-widthnum+1:end,:),dx,dz,dt,Vxy,Dxy);
    [ux3,ux2,ux1,uxsum]=PML(ux3,ux2,ux1,P2(:,end-widthnum+1:end)',dz,dx,dt,Vz,Dz);
    P3(widthnum-5:-1:1,:)=uzsum(6:end,:);
    P3(end-widthnum+6:end,:)=uysum(6:end,:);
    for j=5:widthnum
        P3(1-j+widthnum:end+j-widthnum,N-widthnum+j)=uxsum(j,1-j+widthnum:end+j-widthnum);
    end
    %pcolor(P(:,:,i+1)),shading interp  %
    waitbar(i/T,hwait);
    if any(any(abs(P3)>1e20))
        set(handles.text3,'string',char(get(handles.text3,'string'),'wrong:the matrix may will have NAN! So it has been stopped...'));
           %防止溢出
           issucc=0;
        return;
    end

    if mod(i,5)==0
        P(:,:,i/5)=P3;
    end
    P1=P2;
    P2=P3;
end
bftext=get(handles.text3,'string');
set(handles.text3,'string',char(bftext(1:end-1,:),['running : ',num2str(100),' %']));
close(hwait);
tend=toc(tstart);   %计时器结束
set(handles.text3,'string',char(get(handles.text3,'string'),['run cost time: ',num2str(tend),' s ']));
P=P(widthnum+1:end-widthnum,1:end-widthnum,:);    %删去其pml边界
P(abs(P)<0.5e3)=0;
rho=rho(widthnum+1:end-widthnum,1:end-widthnum);
V=V(widthnum+1:end-widthnum,1:end-widthnum);
K=K(widthnum+1:end-widthnum,1:end-widthnum);
%set(handles.text3,'string',char(get(handles.text3,'string'),'creating the image...'));
%seis_Pplot(P,x_xl,z_xl)
%disp('saving the data...')
%save('wavedata.mat',single(P))
%disp('creating the video...')
%savevideo('seismic simulation',P,x_xl,z_xl,t_xl,T)
set(handles.text3,'string',char(get(handles.text3,'string'),'end of data process'));
issucc=1;
%% end of main
%% 

function d2P=hdif2(P,dim,dx)
C1=[ 1.6667 -0.2381 0.0397 -0.0050 0.0003];  %二阶导数的线性系数
d2P=(diff2(P,dim)*C1(1)+diff2k(P,dim,2)*C1(2)+diff2k(P,dim,3)*C1(3)...
        +diff2k(P,dim,4)*C1(4)+diff2k(P,dim,5)*C1(5))/dx/dx;


function dP=hdif1(P,dim,dx)
C1=[ 1.6667 -0.4762 0.1190 -0.0198 0.0016]/2;  %一阶导数的线性系数
dP=(diff1k(P,dim,1)*C1(1)+diff1k(P,dim,2)*C1(2)+diff1k(P,dim,3)*C1(3)...
        +diff1k(P,dim,4)*C1(4)+diff1k(P,dim,5)*C1(5))/dx;
   
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


function result=input_sig(A,f,t_stid,t_xl)
%点震源信号：RICHER子波
t=t_xl(t_stid:end)-t_xl(t_stid+40);
result=A*exp(-pi*pi*f*f*t.*t).*(1-2*pi*pi*f*f*t.*t);
result=[zeros(1,t_stid-1),result];
 
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

figure(4)
[m,n,T]=size(P);

maxp=5e3;
minp=-5e3;

for i=1:10:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp,view(0,-90);  %wrong 
    colorbar,caxis([minp,maxp]);axis equal;
    colormap();%jet/gray/HSV
    %wait=waitbar(i/100,[num2str(i),'%','  t= ',num2str(t_xl(i))]);
    title('seismic simulation');xlabel('x/m'),ylabel('z/m');
    title(num2str(i));
    pause(0.1);
end

function savevideo(name,P,x_xl,z_xl,t_xl,T)
vidobj=VideoWriter(name);
open(vidobj);
for i=1:T
   %plot
   frame=getframe;
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
for i=n/2:n
    V(:,i)=4;
end
V=V*1e3;



% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global t;global P;global x_xl;global z_xl;global t_xl;
global tid;global mapid;
tid=min(floor(get(handles.slider2,'value')*t/5)+1,t/5);
axes(handles.axes1);
maxp=max(max(P(:,:,tid)));
minp=min(min(P(:,:,tid)));
ul=max(maxp,minp);
pcolor(x_xl,z_xl,(P(:,:,tid)')),shading interp,view(0,-90);  %wrong 
colorbar;axis equal,caxis([-ul,ul]);
    switch mapid
        case 1
            colormap(hsv); %gray /jet / HSV
        case 2
            colormap(jet);
        case 3
            colormap(gray);
    end
title(['T=',num2str(tid*10),' ms']);xlabel('x/m'),ylabel('z/m');

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function get_sig_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to get_sig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function getsig_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to getsig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[x,y]=ginput(1);
global P;global x_xl;global z_xl;global t_xl;
idx=find(x<x_xl, 1 );
idy=find(y<z_xl, 1 );
figure
[~,~,T]=size(P);
sig=reshape(P(idx,idy,:),[1,T]);
plot(t_xl(5:5:end),zeros(1,T),'linewidth',1)
hold on,plot(t_xl(5:5:end),sig,'linewidth',4),
xlabel('time'),ylabel('pressure'),set(gca,'FontSize',16);


% --------------------------------------------------------------------
function surface_signal_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to surface_signal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[~,y]=ginput(1);
global P;global z_xl;global t_xl;global x_xl;
z0=find(y<z_xl, 1 );
figure(6)
[m,~,T]=size(P);
grd=reshape(P(:,z0,:),[m,T]);
 %ma=1e4; %参数可调
pcolor(x_xl,t_xl(5:5:end)',grd'),shading interp,
xlabel('x/m'),ylabel('t/s'),view(0,-90),colorbar;
set(gca,'FontSize',16);
%caxis([-ma,ma]);
colormap(gray);
title(['Depth is ',num2str(z_xl(z0-4)),' m'])


% --------------------------------------------------------------------
function point_data_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to point_data (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
dcm_obj = datacursormode();
set(dcm_obj,'UpdateFcn',@myupdatefcn);


function txt = myupdatefcn(empt,event_obj)
% Customizes text of data tips
global tid;global x_xl;global z_xl;global P;
pos = get(event_obj,'Position');
idx=find(pos(1)<x_xl, 1 );
idy=find(pos(2)<z_xl, 1 );
txt = {['X : ',num2str(pos(1))],...
	      ['Z : ',num2str(pos(2))],['P : ',num2str(single(P(idx,idy,tid)))]};

% --------------------------------------------------------------------
function TDview_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to TDview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t;global P;global x_xl;global z_xl;global t_xl;
global tid;global mapid;

tid=min(floor(get(handles.slider2,'value')*t/5)+1,t/5);
[X,Z]=meshgrid(x_xl,z_xl);
mesh(handles.axes1,X,Z,(P(:,:,tid)')),shading interp; 
colorbar;
    switch mapid
        case 1
            colormap(hsv); %gray /jet / HSV
        case 2
            colormap(jet);
        case 3
            colormap(gray);
    end
title(['T=',num2str(10*(tid)),' ms']);xlabel('x/m'),ylabel('z/m'),zlabel('P');

% --------------------------------------------------------------------
function facepaint_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to facepaint (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3
global mapid;
mapid=get(handles.popupmenu3,'Value');

% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function time2frequent_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to time2frequent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function Pcontour_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to Pcontour (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t;global P;global x_xl;global z_xl;global t_xl;
global tid;
tid=min(floor(get(handles.slider2,'value')*t/5)+1,t/5);
[X,Z]=meshgrid(x_xl,z_xl);
contour(handles.axes1,X,Z,(P(:,:,tid)')),shading interp;view(0,-90); 
colorbar;
title(['T=',num2str(10*(tid)),' ms']);xlabel('x/m'),ylabel('z/m'),zlabel('P');

% --------------------------------------------------------------------
function showinput_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to showinput (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t_xl;global x_xl;global z_xl;
datas=get(handles.uitable2,'data');
A=cell2mat(datas(2,2));
f=cell2mat(datas(1,2));
xid=cell2mat(datas(5,2));
zid=cell2mat(datas(6,2));
t_stid=cell2mat(datas(7,2));
insig=input_sig(A,f,t_stid,t_xl);
figure(9)
subplot(2,1,1),plot(t_xl,insig),xlabel('t/ms'),ylabel('press'),title('input signal');
grid on,set(gca,'FontSize',14);
subplot(2,1,2),plot(abs(fft(insig))),title('| P(w) |');
set(gca,'FontSize',14),grid on;


% --------------------------------------------------------------------
function raypath_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to raypath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global t;global P;global x_xl;global z_xl;global t_xl;
global tid;
%!!!!!!    ????
[m,n,~]=size(P);
A=zeros(m,n);
for i=1:3:t
    id=find(imbinarize(abs(P(:,:,i))));
    A(id)=A(id)-1;
end
A=A(4:24:end,4:24:end);
[fx,fy]=gradient(A);
tid=min(floor(get(handles.slider2,'value')*t)+1,t);
[X,Z]=meshgrid(x_xl,z_xl);
maxp=max(max(P(:,:,tid)));
minp=min(min(P(:,:,tid)));
ul=max(maxp,minp);
pcolor(X,Z,(P(:,:,tid)')),shading interp,axis equal,view(0,-90); 
caxis([-ul,ul]);colorbar;
hold on,quiver(X(4:24:end,4:24:end),Z(4:24:end,4:24:end),fx,fy);
hold off;
title(['T=',num2str(t_xl(tid)),'s  id=',num2str(tid)]);xlabel('x/m'),ylabel('z/m'),zlabel('P');

% --------------------------------------------------------------------
function savevideo_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to savevideo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function helptext_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to helptext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
open('other/help.m');

% --------------------------------------------------------------------
function bigger_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to bigger (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --------------------------------------------------------------------
function binwrite_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to binwrite (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global P;
[filename ,pathname]=uiputfile({'*.bin','bin file'},'save');
str=strcat(pathname,filename);
fid=fopen(str,'wb');
fwrite(fid,P,'single');
fclose(fid);
shelp=['  '];%?????????
fid=fopen('help of SU shell.txt','w');
fwrite(fid,shelp,'single');
fclose(fid);



% --------------------------------------------------------------------
function openmat_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to openmat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global P;global t;global x_xl;global z_xl;global t_xl;
[filename, pathname,index] = ...  
     uigetfile({'*.mat';'*.bin'},'FileSelector'); 
if (index==0)  
    msgbox('您没有选择文件，请重新选择!','打开文件出错','error');  
else  
   Numdata = load(strcat([pathname,filename])); %加载数据,保存为一个结构体  
   dataname=fieldnames(Numdata); %获取结构体的所有变量名，保存为一个矩阵  
   strname = dataname(1,1);   
   P=getfield(Numdata,char(strname));% 要将变量名转换为字符串，getfield获取结构体中指定变量的值，然后用assignin将变量加载到工作空间
   [m,n,t]=size(P);
   x_xl=1:m;
   z_xl=1:n;
   t_xl=1:t;
   msgbox('load success！','确认','warn');  
end  


% --- Executes during object creation, after setting all properties.
function uitable2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uitable2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --------------------------------------------------------------------
function matsave_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to matsave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global m;global n;global t;global P;global x_xl;global z_xl;global t_xl;
global x0id;global z0id;global t0id;global A;
global f0;global width;global depth;
[filename ,pathname,uiid]=uiputfile({'*.mat','MATLAB data file'},'save');
if uiid~=0
str=strcat(pathname,filename);
set(handles.text3,'string',char(get(handles.text3,'string'),'saving the data...'));
save(str,'P','x_xl','z_xl','t_xl','m','n','t','width','depth','A','f0','x0id','z0id','t0id');
set(handles.text3,'string',char(get(handles.text3,'string'),'have save the data...'));
end


% --------------------------------------------------------------------
function savepic_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to savepic (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename ,pathname,uiid]=uiputfile({'*.jpg','picture file'},'save');
if uiid~=0
    str=strcat(pathname,filename);
    h=getframe(handles.axes1);  
    imwrite(h.cdata,str);  
end


% --------------------------------------------------------------------
function loadtobase_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to loadtobase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function VSP_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to VSP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[x,~]=ginput(1);
global P;global x_xl;global t_xl;global z_xl;
idx=find(x<x_xl, 1 );
figure(6)
[~,n,T]=size(P);
grd=reshape(P(idx,:,:),[n,T]);
pcolor(z_xl,t_xl(5:5:end)',grd'),shading interp,
xlabel('z/m'),ylabel('t/s'),view(0,-90),colorbar;
set(gca,'FontSize',16)
colormap(gray);
title(['Offset is ',num2str(x_xl(idx)),' m'])
