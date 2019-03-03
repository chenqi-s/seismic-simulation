function [id,C]=test2(input,k)
%希望一块一块地聚类（携带的x的信息）！！
[id,C]=kmeans(input,k);
j=1;
for i=1:k
    dff=diff(find(id==i));
    idn=find(dff>1);
    if ~isnan(idn) 
        %??/
    end
end