function seisplot(P,T,x_xl,z_xl)
%show the seismic in every step time 
if nargin==1
    [m,n,T]=size(P);
    x_xl=1:m;z_xl=1:n;
elseif nargin==2
    [m,n]=size(P(:,:,1));
    x_xl=1:m;z_xl=1:n;
end
[m,n,t]=size(P);
%P=atan(-P/min(min(min(P)))*180);
% sp=sort(reshape(P(:,:,floor(end*0.2):end),1,m*n*(t-floor(t*0.2)+1)));
% maxp=sp(floor(end*0.999));
% minp=sp(floor(end*0.002));
% maxp=max(max(max(P(:,:,floor(end*0.2):end))));
% minp=min(min(min(P(:,:,floor(end*0.2):end))));
maxp=max(max(max(P)))*0.6;
minp=min(min(min(P)))*0.6;
% minp=-4e3;
% maxp=4e3;
for i=1:T
    pcolor(x_xl,z_xl',(P(:,:,i)')),shading interp,view(0,-90);  %wrong 
    colorbar,caxis([minp,maxp]);axis equal;
    colormap(gray(41)); %gray /jet / HSV
    xlabel('x/m'),ylabel('z/m');
    set(gca,'FontSize',16);
    title(['T= ',num2str(i*10),' ms']);
    pause(0.1);
end