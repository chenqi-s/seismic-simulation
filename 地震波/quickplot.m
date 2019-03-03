function quickplot(P,x_xl,z_xl,T)
ma=max(max(abs(P(:,:,T))));
%ma=0.8e5;
figure,pcolor(x_xl,z_xl',(P(:,:,T)')),shading interp,view(0,-90);  %wrong 
    colorbar,caxis([-ma,ma]);axis equal;
    colormap(hsv(201)); %gray /jet / HSV
    xlabel('x/m'),ylabel('z/m');
    set(gca,'FontSize',16);
    title([' T= ',num2str(T*10),' ms ']);