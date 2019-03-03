function [A,fx,fy]=raypath(P,x_xl,z_xl)
% 获得射线路径，只能对中心点扩散波场做！！
[m,n,t]=size(P);
A=zeros(m,n);
for i=1:t
    id=find(imbinarize(abs(P(:,:,i))));
    A(id)=A(id)-1;
end
A=A(4:24:end,4:24:end);
[X,Z]=meshgrid(x_xl,z_xl);
[fx,fy]=gradient(A);
pcolor(x_xl,z_xl',(P(:,:,30))),shading interp,view(0,-90);  %wrong 
    colorbar;axis equal;
    colormap(gray); %gray /jet / HSV
    xlabel('z/m'),ylabel('x/m');
    set(gca,'FontSize',16);
hold on,
quiver(X(end:-24:4,end:-24:4),Z(end:-24:4,end:-24:4),-fx,-fy),view(90,90);
   axis equal;
