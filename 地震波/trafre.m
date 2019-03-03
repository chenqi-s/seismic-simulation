function F=trafre(A)
% 把P从时间域转化为频域
[m,n,t]=size(A);
F=zeros(m,n,t);
for i=1:m
    for j=1:n
        F(i,j,:)=fft(A(i,j,:));
    end
end