%%SE DEBEN DE CARGAR LOD DATOS EN X E Y
%X E Y DEBEN DE SER UN VECTOR FILA 
clear;clc;
load('Polinomio4.mat','x','y')
%%
%%%%%%VISUALIZAR LA DATA PARA CHECAR CUAL GRADO DE POLINOMIO PODRIA SER
figure
plot(x,y,'ok')
xlabel('x')
ylabel('y')
title('Data Original')
%%
%%%%%INGRESAR LOS DATOS CORRESPONDENTES DE LOS QUE SE DESEA LA REGRESIÓN
%THETA DEBE DE SER UN VECTOR FILA
orden=4;
alpha=0.001;
theta=zeros(1,orden);
b=0;
epochs=90000;
%%
%%%%%%%%%%%%%%%ENTRENAR LA REGRESIÓN%%%%%%%%%%%%%%%%%%
[theta,b,MSE]=Entrenamiento(x,y,theta,b,alpha,orden,epochs);
%%
%%%%%%%%%%%%GRAFICAT ELPOLINOMIO Y COMPARAR%%%%%%%%%%%%
PlotLinear(x,y,theta,b,orden,MSE)
%%
function [THETA,B]=ActualizarPesos(X,Y,THETA,B,ALPHA,ORDEN)
    N=length(X);
    dTheta=zeros(1,ORDEN);
    dB=0;
    f=zeros(1,N);
    for i=1:N
        for j=1:ORDEN
            f(1,i)=f(1,i)+(X(i)^j)*THETA(j);
        end
    end
    f=Y-(f+B);
    for i=1:ORDEN
        dTheta(i)=sum(-2*(X.^i).*f)/N;
    end
    dB=sum(-2*f)/N;
    THETA=THETA-(ALPHA*dTheta);
    B=B-(ALPHA*dB);
end

function Error=MeanSquareError(X,Y,THETA,B,ORDEN)
    N=length(X);
    f=zeros(1,N);
    for i=1:N
        for j=1:ORDEN
            f(1,i)=f(1,i)+(X(i)^j)*THETA(j);
        end
    end
    f=Y-(f+B);
    Error=sum(f.^2)/N;
end

function [THETA,B,VECERROR]=Entrenamiento(X,Y,THETA,B,ALPHA,ORDEN,EPOCHS)
    VECERROR=zeros(1,EPOCHS);
    for k=1:EPOCHS
        [THETA,B]=ActualizarPesos(X,Y,THETA,B,ALPHA,ORDEN);
        VECERROR(k)=MeanSquareError(X,Y,THETA,B,ORDEN);
    end
end

function PlotLinear(X,Y,THETA,B,ORDEN,MSE)
    X2=min(X):0.01:max(X);
    N=length(X2);
    Y2=zeros(1,N);
    for l=1:N
        for h=1:ORDEN
            Y2(l)=Y2(l)+(X2(l)^h)*THETA(h);
        end
    end
    Y2=Y2+B;
    figure
    plot(X,Y,'o')
    hold on
    plot(X2,Y2,'LineWidth',3)
    xlabel('x')
    ylabel('y')
    title('Regresión de grado',ORDEN)
    legend('Datos','Regresión')
    hold off
    figure
    plot(MSE)
    xlabel('Epochs')
    ylabel('Error')
    title('Mean Square Error')
end