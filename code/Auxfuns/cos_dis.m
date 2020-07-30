function A=cos_dis(Z)
%
[~,n]=size(Z);
A=zeros(n,n);
for i=1:n
    for j=i:n
        a=Z(:,i);
        b=Z(:,j);
        A(i,j)=dot(a,b)/(norm(a)*norm(b));
        A(j,i)=dot(a,b)/(norm(a)*norm(b)); 
    end
end







