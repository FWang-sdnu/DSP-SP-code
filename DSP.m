function [ W ] = DSP( X,Xl,A,Y,param,MAX_ITER,islocal)
% function [ W,obj] = DSP( X,Xl,A,Y,param,MAX_ITER,islocal)

%2021.8.25 ��ԭʼ
%m-reduce dimension
% X:d*N
if nargin < 7
    islocal = 1;  %only update the similarities of neighbors
end;

[d,N] = size(X);
Nl = size(Xl,2);
%c = size (Yl,2);
thresh = 1e-11;
%bb = 10;

%calculate U
Ull = param.bb*ones(Nl,1);
Ul = diag(Ull);
Uuu = ones(N-Nl,1);
Uu = diag(Uuu);
U = blkdiag(Ul,Uu);   


%%initialization
S = A;
W = zeros(d,param.m);  %%difference
l = ones(N,1);
% I = eye(d,d);
Hc = eye(N,N)-(l*l')/N;   %N*N
Ha = (X*Hc*X'+eye(d,d))\X*Hc;    %d*N
%Hb = Hc*X'*Ha+(l*l')/N;
Hd = Hc-Hc*X'*Ha;

for iter = 1:MAX_ITER
    %calculate Ls
    Ws = (S'+S)/2;
    Ds = diag(sum(Ws));
    Ls = Ds - Ws;
    
   %update F
    
        F1 = param.alpha*U+2*param.lambda*Ls+param.beta*Hd;
        F = param.alpha*F1\U*Y;

   %update b
        b = (F'*l-W'*X*l)/N;
   
   %update W
        W = Ha*F;
     
   %update S
        dist = L2_distance_1(F',F');
        for i=1:N
        a0 = A(i,:);  
        if islocal == 1   
            idxa0 = find(a0>0);
        else
            idxa0 = 1:num;
        end;
        ai = a0(idxa0);
        di = dist(i,idxa0);
        ad = ai-0.5*param.lambda*di;
        S(i,idxa0) = EProjSimplex_new(ad);
        end;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %calculate obj
        obj1 = norm(S-A,'fro')^2;
        obj2 = 2*param.lambda*trace(F'*Ls*F);
        obj3 = param.beta*(trace((X'*W+l*b'-F)'*(X'*W+l*b'-F))+trace(W'*W));
        obj4 = param.alpha*trace((F-Y)'*U*(F-Y));
      
        obj(iter) = obj1 +obj2 + obj3 + obj4;
        plot(obj);
        if iter>2 && ( obj(iter-1)-obj(iter) )/obj(iter-1) < thresh
             break;
        end
        fprintf('Iter %d\tobj=%f\n',iter,obj(end));
                       
end;   




end

