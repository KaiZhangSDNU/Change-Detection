function [Z,E] = NNLapLRR2(X,L,lambda,beta,gamma)
% This routine solves the following nuclear-norm optimization problem 
% by using inexact Augmented Lagrange Multiplier(iALM(ADM)),
%   min |Z|_* + lambda*|E|_2,1 +beta*|J|_1 +gamma*tr(M*L*M') 
%     s.t. X=XZ+E  
%          Z=J
%          Z=M
% ----------------------------------
% ALM 求解
% inputs:
%     X --    d*N  data matrix: d is the data dimension, and N is the 
%                   number of data vector
%     L --    N*N  Laplician Matiix: L=D-W ()
%   lambda  --  regularized parameters: banlance the error
%    beta   --  regularized parameters: banlance the non-negative
%               constraint
%    gamma  -- regularized parameters: banlance the manifold regularity

% Output:
%     Z -- N*N 
%     E -- error matrix

if nargin<3
    lambda = 1; beta=1; gamma=1;
end

tol = 1e-8;
maxIter = 1e6;
[d, n] = size(X);
rho = 1.1;
max_mu = 1e30;
mu = 0.1;  % 参数有修改
%eta=norm(X).^2+1;
%% Initializing optimization variables
% intialize
Z = zeros(n,n);
M = zeros(n,n);
J = zeros(n,n);
%N = zeros(n,n);
E = sparse(d,n);


Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = zeros(n,n);
%Y4 = zeros(n,n);
%inv_x=inv(2*eye(n)+X'*X);
eta=norm(X).^2;
%% Start main loop
iter = 0;
disp(['initial,rank=' num2str(rank(Z))]);
while iter<maxIter
    iter = iter + 1;
    
% ============== Update N===== SVT decompition
     tempN =Z+X'*(X-X*Z-E+Y1/mu)+(Z-M+Y2/mu+Z-J+Y3/mu )/eta;
    [U,sigma,V] = svd(tempN,'econ');
 %  [U,sigma,V] = lansvd(temp,30,'L');
    sigma = diag(sigma);
    svp = length(find(sigma>1/(mu*eta)));
    if svp>=1
        sigma = sigma(1:svp)-1/(mu*eta);
    else
        svp = 1;
        sigma = 0;
    end
    Z = U(:,1: svp)*diag(sigma)*V(:,1:svp)';
%===============更新Z=================

 % Z=inv_x*((X'*Y1-Y2-Y3)/mu+M+J+X'*X-X'*E);
 
%=============更新J========= Shrinkage Operator
    tempJ = Z + Y2/mu;
    [Uj,sigmaJ,Vj] = svd(tempJ,'econ');
 %  [U,sigma,V] = lansvd(temp,30,'L');
    episilon = beta/mu;  % threshold
    sigmaJ = diag(sigmaJ);
    % 这里可以写个 收缩阈值操作 的算子
    svpBig = find(sigmaJ>episilon);
    svpSmall = find(sigmaJ<-episilon);
    tempS=zeros(length(sigmaJ),1);
    tempS(svpBig,:)=sigmaJ(svpBig,:)-episilon*ones(length(svpBig),1);
    tempS(svpSmall,:)=sigmaJ(svpSmall,:)+episilon*ones(length(svpSmall),1);
%     Svp0=find(tempS<0);
%     tempS(Svp0,:)=0;
    J = Uj*diag(tempS)*Vj';
%     tempZero=J<0;
%     J(tempZero)=0;
%     
%=================更新M============temo
tempInv = inv(2*gamma*L/mu+eye(n));
 M=(Y3/mu+Z)* tempInv;   
    
%=========== 更新E ==========
    xmaz = X-X*Z;
    temp = X-X*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);
 
%  ---------------    %
    leq1 = xmaz-E;
    leq2 = Z-J;
    leq3 = Z-M;
   % leq4 = Z-N;
    stopC1 = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC2 = max(max(abs(leq3)));
    stopC=max(stopC1,stopC2);
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z)) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
      %  Y4 = Y4 + mu*leq4;
        mu = min(max_mu,mu*rho);
    end  
end

%% 子函数
function [E] = solve_l1l2(W,lambda1)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda1);
end


function [x] = solve_l2(w,lambda1)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda1
    x = (nw-lambda1)*w/nw;
else
    x = zeros(length(w),1);
end    
    
    
    
















