clc;
clear;

%% parameter setting
per_ratio=0.01; % The training percentage
img_name='Salinas'; 
num_Pixel=32;  % number of superpixels
par.r=2;       % radius r
random_iters=10; 

%add path
addpath(genpath('./classification_code/'));
addpath('./common/');
addpath('./data/');
addpath('./train_indexes2/');
addpath('./Entropy Rate Superpixel Segmentation/');

%% prepare for data 
%% read file
R=importdata([img_name '_corrected.mat']);
gt=importdata([img_name '_gt.mat']);
R=double(R);
[m,n,d]=size(R);

par.sigma=1.0;
par.D1=m; 
par.D2=n;

data3D=R; 
data3D = data3D./max(data3D(:));

%super-pixels segmentation
labels = cubseg(data3D,num_Pixel);
labels=labels+1;
[sub_I,size_subimage,position_2D]=Partition_into_subimages_superpixel(data3D,labels);

%graph_constraint
X = Unfold(data3D, [m,n,d], 3);
[G1,D1,W_sparse1]=graph_constraint(X,position_2D,par,num_Pixel,labels);

%SGTRPCA
for cnt=1:num_Pixel
    idx = m;
    iend = 0;
    jdx = n;
    jend = 0;
    for i =1:m
        for j = 1:n
            if labels(i,j)==cnt
                jdx = min(j,jdx);
                jend = max(j,jend);
                idx = min(i,idx);
                iend = max(i,iend);
            end
        end
    end
    tmp_position = [idx,iend,jdx,jend];
    position{cnt} = tmp_position;
end	

%Salinas
alpah = 0.1;
beta = 10;
max_iter = 200;
mumax = 1e10;
mu = 1e-4;
tol = 1e-4;
[L,E,Z]=SGTRPCA(data3D,G1,position,num_Pixel,alpah,beta,max_iter,mumax,mu,tol);

[res,accracy_SVM1,TPR_SVM1,Kappa_SVM1,accracy_SVM2,TPR_SVM2,Kappa_SVM2,...
Predict_SVM1,Predict_SVM2,time_result] = my_Classification_V2_CK_ratio_multiple_iters_time(L,data3D,gt,random_iters,img_name,per_ratio);



