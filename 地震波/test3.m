function CA=test3(P,x_xl,z_xl,t_xl)
[m,n,t]=size(P);
A=reshape(P(101:200,5,:),[100,t]);
CA=cumsum(A,1);


