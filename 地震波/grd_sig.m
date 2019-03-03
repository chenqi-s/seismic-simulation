function grd_sig(P,x_xl,z_xl,t_xl,z0)
%在Z＝z0处接受到的（x，t）信号
if nargin==1
    z0=5;
end
[m,~,T]=size(P);
grd=reshape(P(:,z0,:),[m,T]);
 ma=max(max(grd)); %参数可调
 ma=0.8e4;
pcolor(x_xl,t_xl',grd'),shading interp,
xlabel('x/m'),ylabel('t/s'),view(0,-90),colorbar;
set(gca,'FontSize',16);
caxis([-ma,ma]);
colormap(gray(101));
title(['Depth is ',num2str(z_xl(z0-4)),' m'])