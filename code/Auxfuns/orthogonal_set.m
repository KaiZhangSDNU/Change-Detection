function A=orthogonal_set(C,Thre1,Thre2) 
%
A=[];
[~,n]=size(C);
for i=1:n
    index_1=find(C(:,i)<=Thre1); % positive thre
    index_2=find(C(:,i)>=Thre2); % negative thre
    Com_set=intersect(index_1,index_2);
    A=union(A,Com_set); 
end








