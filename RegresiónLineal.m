clear 
clc
df=readmatrix('test.csv');
x=df(:,1);
y=15.2*df(:,2);
w=0.1;
b=0.1;
alpha=0.0001;
epochs=160000;
[w,b,MSE]=LinearRegresion(x,y,w,b,alpha,epochs);
function [W,B]=UpdateWeights(X,Y,W,B,alpha)
    dw=0;
    db=0;
    N=length(X);
    for i=1:N
        f=Y(i)-(W*X(i)+B);
        dw=dw-2*X(i)*f;
        db=db-2*f;
    end
    dw=1/N*dw;
    db=1/N*db;
    W=W-alpha*dw;
    B=B-alpha*db;
end
function Error=MeanSquareError(X,Y,W,B)
    Error=0;
    N=length(X);
    for j=1:N
        Error=Error+(Y(j)-(W*X(j)+B))^2;
    end
end
function [W,B,MSEVect]=LinearRegresion(X,Y,W,B,alpha,epochs)
    l=1;
    for k=1:epochs
        [W,B]=UpdateWeights(X,Y,W,B,alpha);
        if mod(k,400)==0
            Error=MeanSquareError(X,Y,W,B);
            MSEVect(l)=Error;
            l=l+1;
        end
    end
    figure
    x2=linspace(min(X),max(X),100);
    plot(X,Y,'ob')
    hold on
    plot(x2,W*x2+B,'r')
    xlabel('x')
    ylabel('y')
    title('Data')
    legend('Real','Regresión')
    hold off
    figure
    plot(MSEVect)
    xlabel('Epochs')
    ylabel('MSE')
    title('Mean Square Error')
end