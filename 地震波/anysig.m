function sig=anysig(P,x,z)
[~,~,T]=size(P);
sig=reshape(P(x,z,:),[1,T]);
figure
plot(1:T,zeros(1,T),'linewidth',1)
hold on,plot(1:T,sig,'linewidth',4),
%plot(1:T,smooth(sig,2),'linewidth',2);
xlabel('time'),ylabel('pressure'),set(gca,'FontSize',16);