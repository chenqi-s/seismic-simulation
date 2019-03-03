% F=trafre(P);
% abf=abs(F);
% agf=angle(F);
% %fin=fft(input(2:end));
% abfin=abs(fin);
% agfin=angle(fin);
%[A,~,r]=atten(abf,[110 100],[180 100],[100 100],x_xl(2)-x_xl(1));

%%
% subplot(3,1,2),plot(abfin,'linewidth',2.5),title('fre Amp of input '),xlabel('f'),ylabel('Amplitude')
% set(gca,'FontSize',14),grid on;
% subplot(3,1,1),plot(t_xl,input,'linewidth',2.5),title('input signal'),xlabel('t/s'),ylabel('press')
% set(gca,'FontSize',14),grid on;
% subplot(3,1,3),plot(agfin,'linewidth',2.5),title('fre phase of input '),xlabel('f'),ylabel('phase')
% set(gca,'FontSize',14),grid on;
%%
for i=1:4
    ft=reshape(abf(100,100+20*i,:),[1,360]);
    pt=reshape(P(100,100+20*i,:),[1,360]);
   
subplot(2,1,2),hold on,plot(ft,'linewidth',2.5),title('fre Amp '),xlabel('f'),ylabel('Amplitude')
set(gca,'FontSize',14),grid on;
subplot(2,1,1),hold on,plot(t_xl,pt,'linewidth',2.5),xlabel('t/s'),ylabel('press')
set(gca,'FontSize',14),grid on;

end
%%

function [A,D,r,dsmd]=atten(p,st,en,o,dx)
if st(1)~=en(1)
    x=linspace(st(1),en(1),en(1)-st(1)+1);
    y=round((en(2)-st(2))/(en(1)-st(1))*(x-st(1))+st(2));
else
    y=st(2):en(2);
    x=st(1)*ones(1,length(y));
end
r=sqrt((x-o(1)).^2+(y-o(2)).^2)*dx;
A=zeros(1,length(x));
I=A;D=A;
for i=1:length(x)
    [~,I(i)]=max(diff(smooth(p(x(i),y(i),:),1),2));
    [A(i),D(i)]=max(abs(p(x(i),y(i),max(I(i)-20,1):min(I(i)+20,end))));
end
D=D+I-20;
figure,
plot(r,A,'linewidth',4),hold on,
plot(r,1./sqrt(r)*A(1)*sqrt(r(1)),'linewidth',2.5),legend('simulation Amp with R ','theory A with R ');
hold off,
title('Compare of | P(w) | with R'),xlabel('R/m'),ylabel('Amplitude')
set(gca,'FontSize',14);
figure,
plot(r,D,'linewidth',3),
title('delay phase with R'),xlabel('R/m'),ylabel('delay phase')
set(gca,'FontSize',14);
% figure,
% dsmd=diff(D);
% plot(r(1:end-1),dsmd,'linewidth',3),
% title('d(phase)/dr with R'),xlabel('R (dis of id)'),ylabel('V')
% set(gca,'FontSize',14);

end