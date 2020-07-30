clear all
close all
addpath('data');
addpath('ers-master');
addpath('Auxfuns');
addpath('Gabor');
addpath('LRR');

T1=imread('124033_1_20030430.tif'); % Time 1
T2=imread('124033_1_20090406.tif'); % Time 2
Ind_band=[1,2,3,4,5,7];

DI=double(T2(:,:,Ind_band))-double(T1);

[m,n,b]=size(DI);
DI_1=sum(DI,3)/length(Ind_band);

DI_2 = (DI_1-min(min(DI_1)))/(max(max(DI_1))-min(min(DI_1)));
DI_2 = im2uint8(DI_2);

nC = 1200; 
lambda_prime = 0.5;
sigma = 5.0;
conn8 = 1; 
img = DI_1;
[label] = mex_ers(double(img),nC);
[bmap] = seg2bmap(label,size(img,1),size(img,2)); 
label=label+1;
load('featureVector.mat','featureVector');
X_s=zeros(24,nC); %
Seg_map=reshape(label,[m*n,1]);
DI_column=double(reshape(DI,[m*n,b]));
featureVector = double(featureVector);

for i=1:nC
    index_s=find(Seg_map==i);
    temp1=DI_column(index_s,:);
    temp2=featureVector(index_s,:);
    
    X_s(1:6,i)=sum(temp1,1)'/length(index_s);
    X_s(7:24,i)=sum(temp2,1)'/length(index_s);
end

load('Z.mat','Z');
% % Low-rank representation
% lambda = 10000;
% [Z,E] = lrra(full(X_s),full(X_s),lambda,true);
% figure,imshow(Z,[]);
penalty1=1.2;
penalty2=0.7;
Norm2=zeros(1,nC);
for i=1:nC
    Norm2(i)=norm(Z(i,:),2);
end
Norm2_sum=sum(Norm2);
Z_c_index=[];
for i=1:nC
    if Norm2(i)>=penalty1*((Norm2_sum-Norm2(i))/(nC-1))
        Z_c_index=[Z_c_index,i];
    end
end
Diff_index=setdiff(1:nC,Z_c_index);
Norm2_sub=Norm2(Diff_index);
Z_un_index=[];
for i=1:length(Diff_index)
    if Norm2_sub(i)<=penalty2*((sum(Norm2_sub)-Norm2_sub(i))/(length(Diff_index)-1))
        Z_un_index=[Z_un_index,Diff_index(i)];
    end
end

X_c=X_s(:,Z_c_index);
lambda = 10000;
[Z_c,E_c] = lrra(full(X_c),full(X_c),lambda,true);
Norm2 = zeros(1,size(Z_c,2));
for i = 1:size(Z_c,2)
    Norm2(i) = norm(Z_c(i,:),2);
end
Norm2_sum = sum(Norm2);
X_c_index = [];
penalty3 = 0.8;
for i = 1:size(Z_c,2)
    if Norm2(i)<=penalty3*(Norm2_sum-Norm2(i))/(size(Z_c,2)-1)
        X_c_index = [X_c_index,i];
    end
end
X_c=X_c(:,X_c_index);
X_un=X_s(:,Z_un_index);
X_c=[];
X_un=[];
number_un=length(Z_un_index);
number_c=length(X_c_index);
for i = 1:number_c
    X_cc=[];
    index_c = find(Seg_map==X_c_index(i));
    temp1=DI_column(index_c,:);
    temp2=featureVector(index_c,:);
    
    X_cc(1:6,:)=temp1';
    X_cc(7:24,:)=temp2';
    X_c = [X_c,X_cc];
end
for i = 1:number_un
    X_unun=[];
    index_un = find(Seg_map==Z_un_index(i));
    temp1=DI_column(index_un,:);
    temp2=featureVector(index_un,:);
    
    X_unun(1:6,:)=temp1';
    X_unun(7:24,:)=temp2';
    X_un = [X_un,X_unun];
end

[Dictionary] =PCA(X_c');
D_c = Dictionary;
[Dictionary] =PCA(X_un');
D_un = Dictionary;
num_atom = 24;
w=3; 
f=ones(w,w)/(w*w);
DI_w=convn(DI,f,'same');
gaborfeature_w=convn(reshape(featureVector,[m,n,18]),f,'same');
DI_w_column=double(reshape(DI_w,[m*n,b]));
gaborfeature_w_column=double(reshape(gaborfeature_w,[m*n,18]));
X_w=[DI_w_column'; gaborfeature_w_column'];

lambda_G=10000;
e1=ones(num_atom,1);
I=eye(num_atom,num_atom);
G_un=zeros(size(D_un,2),m*n);
G_c=zeros(size(D_c,2),m*n);
for i=1:m*n
    D_un_X=D_un-repmat(X_w(:,i),[1,size(D_un,2)]);
    A=inv(lambda_G*I+(D_un_X')*D_un_X);
    cons_un=(e1')*A*e1;
    gamma_un=A*e1/cons_un;
    G_un(:,i)=gamma_un;      
    
    D_c_X=D_c-repmat(X_w(:,i),[1,size(D_c,2)]);
    B=inv(lambda_G*I+(D_c_X')*D_c_X);
    cons_c=(e1')*B*e1;
    gamma_c=B*e1/cons_c;
    G_c(:,i)=gamma_c;
end

Recon_un=D_un*G_un;
Recon_c=D_c*G_c;
Err_un=X_w-Recon_un;
Err_c=X_w-Recon_c;
Err_un_col=zeros(1,m*n);
Err_c_col=zeros(1,m*n);
for i=1:m*n
    Err_un_col(i)=norm(Err_un(:,i),2);
    Err_c_col(i)=norm(Err_c(:,i),2);
end
Err_un_map=reshape(Err_un_col',[m,n]);
Err_c_map=reshape(Err_c_col',[m,n]);

Change_map=Err_un_map-Err_c_map;
Change_map(Change_map>0)=1;
Change_map(Change_map<=0)=0;
figure, imshow(Change_map,[]);


