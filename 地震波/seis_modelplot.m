function seis_modelplot(rho,V,K,z_xl,x_xl)
% 地质模型画图
figure(1)
pcolor(x_xl,z_xl',rho'),view(0,-90),shading interp,c=colorbar('southoutside');
c.Label.String = 'kg/m3';xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the density model of seismic simulation')
figure(2)
pcolor(x_xl,z_xl',V'),view(0,-90),shading interp,c=colorbar('southoutside');
c.Label.String = 'm/s';xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the velocity model of seismic simulation')
figure(3)
pcolor(x_xl,z_xl',K'),view(0,-90),shading interp,colorbar('southoutside');
xlabel('x/m'),ylabel('z/m');
set(gca,'FontSize',16);
title('the K model of seismic simulation')
