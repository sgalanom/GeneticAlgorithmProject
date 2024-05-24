syms f(u1,u2)

f(u1,u2) = sin(u1+u2) * sin(u2^2);

psize = 100;    
cross_chrom = 30;    
mutated_chrom= 33;
generations = 1000;

u1_limits = [-1, 2];
u2_limits = [-2, 1]; 

X = zeros(256,2);
for i = 1:16:256
    X(i:i+15,1) = linspace(-1,2,16);
end
temp = 1;
for i = 0:15
    X(i+temp:temp+15+i,2) = -2 + i*(0.2);
    temp = temp + 15;
end
y_test = zeros(256,1);
for i = 1:256
    y_test(i,1) = f(X(i,1),X(i,2));
end

P = population(psize);

for i=1:generations
    Cr = crossover(P,cross_chrom);
    Mu = mutation(P,mutated_chrom);
    P(psize+1:psize+2*cross_chrom,:) = Cr;
    P(psize+2*cross_chrom+1:psize+2*cross_chrom+mutated_chrom,:) = Mu;
    E = evaluation(P,X,y_test);
    P = selection(P,E,psize);
    error_history(i) = min(E);

end

solution = P(1,:); 

a = zeros(15,1); c1 = zeros(15,1); c2 = zeros(15,1);
s1 = zeros(15,1); s2 = zeros(15,1); b = 0;
k1 = 1; k2 = 1;
for i = 1:15
    
    a(i) = bi2de(P(1,k1:(k1+4)),'left-msb');
    c1(i) = bi2de(P(1,(k1+75):(k1+75+4)),'left-msb');
    c2(i) = bi2de(P(1,(k1+150):(k1+150+4)),'left-msb');
    s1(i) = bi2de(P(1,(k2+225):(k2+225+2)),'left-msb');
    s2(i) = bi2de(P(1,(k2+270):(k2+270+2)),'left-msb');
    k1 = k1 + 5;
    k2 = k2 + 3;
       
    a(i) = (a(i)/10) - 1.5;
    c1(i) = (c1(i)/10) - 1;
    c2(i) = (c2(i)/10) - 2;
    s1(i) = (s1(i)/10) + 0.1;
    s2(i) = (s2(i)/10) + 0.1;       
end

y_hat = zeros(length(y_test),1);
for k = 1:length(y_test)
    y_hat(k) = f_hat(a,b,[c1 c2 s1 s2],X(k,1),X(k,2));
end
mse_value = immse(y_test, y_hat);
mae_value = mean(abs(y_test - y_hat));

disp('Optimal Parameters:');
disp(['a: ', num2str(a')]);
disp(['b: ', num2str(b)]);
disp(['c1: ', num2str(c1')]);
disp(['c2: ', num2str(c2')]);
disp(['s1: ', num2str(s1')]);
disp(['s2: ', num2str(s2')]);
disp(['MSE: ', num2str(mse_value)]);
disp(['MAE: ', num2str(mae_value)]);

figure(1);
f1 = @(x,y) sin(x+y)*sin(y^2);
fsurf(f1,[-1 2 -2 1]);
title('Original Function');
xlabel('u1');
ylabel('u2');

figure(2);
f_p = @(u1,u2) a(1)*exp(-((u1-c1(1))^2/(2*s1(1)^2)+(u2-c2(1))^2/(2*s2(1)^2))) + a(2)*exp(-((u1-c1(2))^2/(2*s1(2)^2)+(u2-c2(2))^2/(2*s2(2)^2))) + a(3)*exp(-((u1-c1(3))^2/(2*s1(3)^2)+(u2-c2(3))^2/(2*s2(3)^2))) + a(4)*exp(-((u1-c1(4))^2/(2*s1(4)^2)+(u2-c2(4))^2/(2*s2(4)^2))) +a(5)*exp(-((u1-c1(5))^2/(2*s1(5)^2)+(u2-c2(5))^2/(2*s2(5)^2))) + a(6)*exp(-((u1-c1(6))^2/(2*s1(6)^2)+(u2-c2(6))^2/(2*s2(6)^2))) + a(7)*exp(-((u1-c1(7))^2/(2*s1(7)^2)+(u2-c2(7))^2/(2*s2(7)^2))) + a(8)*exp(-((u1-c1(8))^2/(2*s1(8)^2)+(u2-c2(8))^2/(2*s2(8)^2))) + a(9)*exp(-((u1-c1(9))^2/(2*s1(9)^2)+(u2-c2(9))^2/(2*s2(9)^2))) + a(10)*exp(-((u1-c1(10))^2/(2*s1(10)^2)+(u2-c2(10))^2/(2*s2(10)^2))) + a(11)*exp(-((u1-c1(11))^2/(2*s1(11)^2)+(u2-c2(11))^2/(2*s2(11)^2))) + a(12)*exp(-((u1-c1(12))^2/(2*s1(12)^2)+(u2-c2(12))^2/(2*s2(12)^2))) + a(13)*exp(-((u1-c1(13))^2/(2*s1(13)^2)+(u2-c2(13))^2/(2*s2(13)^2))) + a(14)*exp(-((u1-c1(14))^2/(2*s1(14)^2)+(u2-c2(14))^2/(2*s2(14)^2))) + a(15)*exp(-((u1-c1(15))^2/(2*s1(15)^2)+(u2-c2(15))^2/(2*s2(15)^2)));
fsurf(f_p,[-1 2 -2 1]);
title('Approximated Function');
xlabel('u1');
ylabel('u2');

figure(3);
subplot(2, 3, 1);
plot(a);
title('a');
subplot(2, 3, 2);
plot(c1);
title('c1');
subplot(2, 3, 3);
plot(c2);
title('c2');
subplot(2, 3, 4);
plot(s1);
title('s1');
subplot(2, 3, 5);
plot(s2);
title('s2');
subplot(2, 3, 6);
plot(b);
title('b');

figure(4);
plot(1:generations, error_history);
title('Error diagram');
xlabel('Generation');
ylabel('Error');

load handel.mat; 
sound(y, Fs);


function [G] = Gaussian(u1,u2,const)
c1 = const(1);
c2 = const(2);
s1 = const(3);
s2 = const(4);
temp = ((u1-c1)^2)/(2*s1^2) + ((u2-c2)^2)/(2*s2^2);
G = exp(-temp);
end


function Y = crossover(P,n)
[x1, y1] = size(P);
Z = zeros(2*n,y1);
for i = 1:n
    r1 = randi(x1,1,2);
    while r1(1)==r1(2)
        r1 = randi(x1,1,2);
    end
    A1 = P(r1(1),:);
    A2 = P(r1(2),:);
    r2 = 1 + randi(y1-1);
    B1 = A1(1,r2:y1);
    A1(1,r2:y1) = A2(1,r2:320);
    A2(1,r2:320) = B1;
    Z(2*i-1,:) = A1;
    Z(2*i,:) = A2;
end
Y = Z;
end


function Y = evaluation(P, u_test, y_test)
[x1, ~] = size(P);
H = zeros(1,x1);
for i = 1:x1
    
    a = zeros(15,1); c1 = zeros(15,1); c2 = zeros(15,1);
    s1 = zeros(15,1); s2 = zeros(15,1); b = 0;
    temp1 = 1; temp2 = 1;
    for j = 1:15
       
        a(j) = bi2de(P(i,temp1:(temp1+4)),'left-msb');
        c1(j) = bi2de(P(i,(temp1+75):(temp1+75+4)),'left-msb');
        c2(j) = bi2de(P(i,(temp1+150):(temp1+150+4)),'left-msb');
        s1(j) = bi2de(P(i,(temp2+225):(temp2+225+2)),'left-msb');
        s2(j) = bi2de(P(i,(temp2+270):(temp2+270+2)),'left-msb');
        temp1 = temp1 + 5;
        temp2 = temp2 + 3;
       
        a(j) = (a(j)/10) - 1.5;
        c1(j) = (c1(j)/10) - 1;
        c2(j) = (c2(j)/10) - 2;
        s1(j) = (s1(j)/10) + 0.1;
        s2(j) = (s2(j)/10) + 0.1;
        
    end
    b = bi2de(P(i,316:320),'left-msb');
    b = (b/10) - 1.5;
    const = [c1 c2 s1 s2];
    
    y_hat = zeros(length(y_test),1);
    for k = 1:length(y_test)
        y_hat(k) = f_hat(a,b,const,u_test(k,1),u_test(k,2));
    end
    H(1,i) = immse(y_test,y_hat);
end
Y = H;
end


function [f_hat] = f_hat(a,b,const,u1,u2)
f_hat = 0;
for i=1:15
    f_hat = f_hat + a(i) * Gaussian(u1,u2,const(i,:));
end
f_hat = f_hat + b;
end


function Y = mutation(P,n)
[x1, y1] = size(P);
Z = zeros(n,y1);
for i = 1:n
    r1 = randi(x1);
    A1 = P(r1,:);       
    r2 = randi(y1);
    if A1(1,r2) == 1
        A1(1,r2) = 0;   
    else
        A1(1,r2) = 1;
    end
    Z(i,:) = A1;
end
Y = Z; 
end


function P = population(n)
P = round(rand(n,320));
end


function [P_new] = selection(P,E,p)
[~, y] = size(P);
Y1 = zeros(p,y);
elite_selection = 3;
for i = 1:elite_selection
    [~,c1] = find(E==min(E));
    Y1(i,:) = P(min(c1),:);
    P(min(c1),:) = [];
    E(:,min(c1)) = [];
end
D = E/sum(E);
E1 = cumsum(D);
N = rand(1);
d1 = 1;
d2 = elite_selection;
while d2 <= p-elite_selection
    if N <= E1(d1)
        Y1(d2+1,:)=P(d1,:);
        N = rand(1);
        d2 = d2 + 1;
        d1 = 1;
    else
        d1 = d1 + 1;
    end
end
P_new = Y1;
end