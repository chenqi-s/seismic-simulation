function [A,D,r,dsmd]=atten(p,st,en,o)
if st(1)~=en(1)
    x=linspace(st(1),en(1),en(1)-st(1)+1);
    y=round((en(2)-st(2))/(en(1)-st(1))*(x-st(1))+st(2));
else
    y=st(2):en(2);
    x=st(1)*ones(1,length(y));
end
r=sqrt((x-o(1)).^2+(y-o(2)).^2);
A=zeros(1,length(x));
I=A;D=A;
for i=1:length(x)
    [~,I(i)]=max(diff(smooth(p(x(i),y(i),:),10),2));
    [A(i),D(i)]=max(abs(p(x(i),y(i),max(I(i)-10,1):min(I(i)+10,end))));
end
D=D+I-10;
figure,
plot(r,smooth(A,20),'linewidth',4),hold on,
plot(r,1./sqrt(r)*A(1),'linewidth',2.5),legend('sim Amp with R ','theory A with R ');
hold off,
title('Compare of Amp with R'),xlabel('R (dis of id)'),ylabel('Amplitude')
set(gca,'FontSize',14);
figure,
smd=smooth(D,10);
plot(r,smd,'linewidth',3),
title('delay phase with R'),xlabel('R (dis of id)'),ylabel('delay phase (of id)')
set(gca,'FontSize',14);
figure,
dsmd=diff(smd);
plot(r(1:end-1),dsmd,'linewidth',3),
title('d(phase)/dr with R'),xlabel('R (dis of id)'),ylabel('V')
set(gca,'FontSize',14);