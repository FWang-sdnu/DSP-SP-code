clear all;
clc;

% load('MSRA25_uni.mat');n =size(X,1);
%  load('Coil20Data_25_uni');n =size(X,1);
%    load('ORL_32x32');n =size(X,1);
% load('USPSdata_20_uni');n =size(X,1);
load('dig1-10_uni');n =size(X,1);Y = Y+1;
% load('CNAE-9');n =size(X,1);


%% PCA
options = [];
options.PCARatio = 0.95;
[eigvector,eigvalue,meanData,new_data] = PCA(X,options);
X = new_data;

%% �������ݼ�
label = unique (Y);
nlabel = histc (Y,label);%ÿ�����������?
c =length(label);

Result = zeros (1,20);
for iter = 1 :20
%% 1.����������ݼ�?
Xl = [];
Yl = [];
Xu = [];
Yu = [];
Xtest = [];
Ytest = [];

rowrank = randperm (size (X,1));
XX = X(rowrank,:);
YY = Y(rowrank,:);


for i = 1:c
    %set labeled samples    
    p = 2;                                          
    nn = ceil(nlabel(i)/2);

    [id,~] = find(YY==label(i));%�ҵ����ڵ�i������������ı��
    YYY = YY(id,:);  %YYY�а������е�i�������?
    XXX = XX(id,:);
    if p == 1
       Xl = [Xl; XXX(1,:)];
       Yl = [Yl;YYY(1,:)];
       Xu = [Xu;XXX(p+1:nn,:)];
       Yu = [Yu;YYY(p+1:nn,:)];
    elseif p+1==nn
        Xl = [Xl; XXX(1:p,:)];
        Yl = [Yl;YYY(1:p,:)];
        Xu = [Xu;XXX(nn,:)];
        Yu = [Yu;YYY(nn,:)];
    else
    Xl = [Xl; XXX(1:p,:)];
    Yl = [Yl;YYY(1:p,:)];
    Xu = [Xu;XXX(p+1:nn,:)];
    Yu = [Yu;YYY(p+1:nn,:)];
    end
    
    %% 2.test part ����
    Xtest = [Xtest;XXX(nn+1:nlabel(i),:)];
    Ytest = [Ytest;YYY(nn+1:nlabel(i),:)]; 
            
end
      Xl=NormalizeFea(Xl);
      Xu = NormalizeFea(Xu);
      Xtest= NormalizeFea(Xtest);
      Xtrain = [Xl;Xu];
      Ytrain = [Yl;Yu];  %һ��
    
%==========================
    Nl = size(Xl,1);
    Nu = size(Xu,1);
    NN = Nl+Nu;
    
    YYY = zeros(NN,c);
    YYl = zeros(Nl,c);
    YYu = zeros(Nu,c);
    YYtrain = zeros(NN,c);
    YYtest = zeros(n-NN,c);
for k = 1:n
   YYY(k,Y(k,1)) = 1;
end
for kk = 1:Nl
    YYl(kk,Yl(kk,1)) = 1;
end
if Nu ==1
    YYu(1,Yu(1,1)) = 1;
else
    for kkk = 1:Nu
       YYu(kkk,Yu(kkk,1)) = 1;
    end
end  
for t =1:NN
    YYtrain(t,Ytrain(t,1)) = 1;
end
for tt = 1: n-NN
    YYtest(tt,Ytest(tt,1)) = 1;
end
    
%% ����������
A = preA(Xtrain',Xl,Yl, 10, 0,p);
% lll=sum(A,2);
MAX_ITER = 20;
best_result = 0;
param.m = c;         %%%reduced dimension
lambda = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5,1e6];
alpha = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5,1e6];
beta = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5,1e6];
bbb = [1,10,1e2,1e3,1e4];

for la = 1:13
   for al = 1:13
         for be = 1:13
            for BB = 1:5
            param.lambda = lambda(la);
            param.alpha = alpha(al);
            param.beta = beta(be);
            param.bb = bbb(BB);
[ W ] = DSP( Xtrain',Xl',A,YYtrain,param,MAX_ITER );

%%  1.unlabel
      Xlw = Xl*W; 
      Xuw = Xu*W;
      rowrank = randperm(size(Xlw,1));
      XX = Xlw(rowrank,:); 
      YY = Yl(rowrank,:);

      row = randperm (size(Xuw,1));
      xx = Xuw(row,:);
      yy = Yu (row,:); 
      y = KNN(xx', XX', YY', 1);
      preY =y';            
      rca = 0;
      Nu = size(Xu,1); 
      for u = 1:Nu
%          if preY (u,:) == Yu(u,:);
         if preY (u,:) == yy(u,:);
             rca = rca+1;
         end
      end
      result= rca/Nu;  


%%  2.test
% % 
%       Xlw = Xl*W; 
%       Xtestw =Xtest*W;
%       rowrank = randperm(size(Xlw,1));
%       XX = Xlw(rowrank,:); 
%       YY = Yl(rowrank,:);
%     
%       row = randperm (size(Xtestw,1));
%       xx = Xtestw(row,:);
%       yy = Ytest (row,:);
%       y = KNN(xx', XX', YY', 1);  
%       preY =y';            
%       rca = 0;
%       Nu = size(Xtest,1);
%       for u = 1:Nu
%          if preY (u,:) == yy(u,:);
%              rca = rca+1;
%          end
%      end
%      result= rca/Nu;  

%%

if  result>best_result;
    best_result=result;
    bestlambda = lambda(la);
    bestalpha = alpha(al);
    bestbeta = beta(be);
end
            end 
        end
    end
end



 Result(1,iter) = best_result;

end
mean_result = mean(Result);
std_result = std(Result);















    