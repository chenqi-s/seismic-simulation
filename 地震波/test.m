function  x=test(in)
global input;
input=in;
xx=1:length(input);
x=ga(@(x) dis(x),4,[],[],[],[],[0;-inf;0;0],[inf;inf;length(xx);length(xx)]);
plot(x(1)*heaviside(xx-x(3))+x(2)*heaviside(xx-x(4)));
hold on,plot(input)

function R=dis(x)
global input;
xx=1:length(input);
R=sum((input-x(1)*heaviside(xx-x(3))-x(2)*heaviside(xx-x(4))).^2);


