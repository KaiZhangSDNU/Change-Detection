function W = Construct_affinitymatrix(G_sparse,k)
%  k nearest neighbor (kNN) graph 
% data matrix: G
% number of neighbors: k
G=full(G_sparse);
[~,n]=size(G);
W=zeros(n,n);

for i=1:n
    G_i=repmat(G(:,i),[1,n]);
    G_i_err=G_i-G;
    A=G_i_err.*G_i_err;
    norm2_column=sum(A,1);
    
    [~,index_k]=sort(norm2_column);
    temp1=index_k(2:k+1);
    W(i,temp1)=1;    
end






