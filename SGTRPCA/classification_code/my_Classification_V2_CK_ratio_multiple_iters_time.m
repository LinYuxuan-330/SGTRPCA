function [res,accracy_SVM1,TPR_SVM1,Kappa_SVM1,accracy_SVM2,TPR_SVM2,Kappa_SVM2,Predict_SVM1,Predict_SVM2,time_result] = my_Classification_V2_CK_ratio_multiple_iters_time(org_data,re_data,GT_map,random_iters,img_name,ratio)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function: compare the classification performance on org_data and re_data using SVM and SVM_CK classifier
%% Input:
    %% org_data: original hyperspectral image
    %% re_data: processed hyperspectral image
    %% GT_map: classification Ground-truth map
    %% iter: randomly run 'iter' times
%% Output:
    %% OA_SVM1: average Overall accuracy by SVM classifier on org_data;
    %% OA_SVM2 average Overall accuracy by SVM classifier on re_data;
    %% AA_SVM1: average Average accuracy by SVM classifier on org_data;
    %% AA_SVM2 average Average accuracy by SVM classifier on re_data;
    %% ave_Kappa_SVM1: average Kappa coefficient by SVM classifier on org_data;
    %% ave_Kappa_SVM2 average Kappa coefficient by SVM classifier on re_data;
    %% ave_TPR_SVM1: average accuracy of every class by SVM classifier on org_data;
    %% ave_TPR_SVM2 average accuracy of every class by SVM classifier on re_data;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_start=tic;
[m n d] = size(org_data);
Data_R1 = reshape(org_data,m*n,d);
Data_R2 = reshape(re_data,m*n,d);
%% 训练集的10%取样
% CTrain = [6 144 84 24 50 75 3 49 2 97 247 62 22 130 38 10]; %CTrain=[23 89 73 66 71 81 14 71 10 76 112 68 67 89 68 47];%% for indian pines dataset
% CTrain = 50*ones(1,9); % for Pavin
% CTrain = [68 67 67 67 67 68 68 70 69 67 67 67 68 67 68 68]; % for Salinas
%% 初始化多次测试（20次）的结果统计数组
accracy_SVM1 = [];
accracy_SVM2 = [];
accracy_SVMCK1 = [];
accracy_SVMCK2 = [];
Kappa_SVM1 = [];
Kappa_SVM2 = [];
%% 
Predict_SVM1=[];
Predict_SVM2=[];
%%
Kappa_SVMCK1 = [];
Kappa_SVMCK2 = [];
TPR_SVM1 = [];
TPR_SVM2 = [];
TPR_SVMCK1 = [];
TPR_SVMCK2 = [];
%%
Predict_SVMCK1=[];
Predict_SVMCK2=[];
%% 重复测试
time_end1=toc(time_start);
for ith_iter=1:random_iters
    %% 训练集的样本位置及测试集样本位置
%     [loc_train, loc_test, CTest] = Generating_training_testing(GT_map,CTrain);
	time_start2(ith_iter)=tic;
    Train_Test=load_loc_train_test_ratio(img_name,ith_iter,ratio);
    loc_train=Train_Test.loc_train;
    loc_test=Train_Test.loc_test;
    CTest=Train_Test.CTest;
    CTrain=Train_Test.CTrain;
    %% SVM
    [accur11, Kappa11,TPR11,Predict11] = Excute_SVM2(Data_R1, loc_train, CTrain, loc_test, CTest);
    accracy_SVM1 = [accracy_SVM1 accur11];
    TPR_SVM1 = [TPR_SVM1;TPR11];
    Kappa_SVM1 = [Kappa_SVM1 Kappa11];
    Predict_SVM1= [Predict_SVM1 Predict11];
    
    [accur12, Kappa12,TPR12,Predict12] = Excute_SVM2(Data_R2, loc_train, CTrain, loc_test, CTest);
    accracy_SVM2 = [accracy_SVM2 accur12];
    TPR_SVM2 = [TPR_SVM2;TPR12];
    Kappa_SVM2 = [Kappa_SVM2 Kappa12];
    Predict_SVM2= [Predict_SVM2 Predict12];

    % %% SVMCK
    % [accur21, Kappa21,TPR21,Predict21]  = Excute_SVMCK2(org_data, loc_train, CTrain, loc_test, CTest, 5, GT_map);
    % accracy_SVMCK1 = [accracy_SVMCK1 accur21];
    % TPR_SVMCK1 = [TPR_SVMCK1;TPR21];
    % Kappa_SVMCK1 = [Kappa_SVMCK1 Kappa21];
    % Predict_SVMCK1= [Predict_SVMCK1 Predict21];

    
    % [accur22, Kappa22,TPR22,Predict22]  = Excute_SVMCK2(re_data, loc_train, CTrain, loc_test, CTest, 5, GT_map);
    % accracy_SVMCK2 = [accracy_SVMCK2 accur22];
    % TPR_SVMCK2 = [TPR_SVMCK2;TPR22];
    % Kappa_SVMCK2 = [Kappa_SVMCK2 Kappa22];
    % Predict_SVMCK2=[Predict_SVMCK2 Predict22];
     time_end2(ith_iter)=toc(time_start2(ith_iter));
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_start3=tic;
[res.ave_OA_SVM1 res.OA_SVM1_std]=mean_std(accracy_SVM1);
[res.ave_OA_SVM2 res.OA_SVM2_std]=mean_std(accracy_SVM2);

[res.ave_Kappa_SVM1	res.Kappa_SVM1_std]=mean_std(Kappa_SVM1);
[res.ave_Kappa_SVM2	res.Kappa_SVM2_std]=mean_std(Kappa_SVM2);

[res.ave_TPR_SVM1 res.TPR_SVM1_std]=mean_stds_1(TPR_SVM1);
[res.ave_TPR_SVM2 res.TPR_SVM2_std]=mean_stds_1(TPR_SVM2);

[res.ave_AA_SVM1 res.AA_SVM1_std]= mean_stds_2(TPR_SVM1);
[res.ave_AA_SVM2 res.AA_SVM2_std]= mean_stds_2(TPR_SVM2);

% [res.ave_accracy_SVMCK1 res.accracy_SVMCK1_std]=mean_std(accracy_SVMCK1);
% [res.ave_accracy_SVMCK2 res.accracy_SVMCK2_std]=mean_std(accracy_SVMCK2);

% [res.ave_Kappa_SVMCK1 res.Kappa_SVMCK1_std]=mean_std(Kappa_SVMCK1);
% [res.ave_Kappa_SVMCK2 res.Kappa_SVMCK2_std]=mean_std(Kappa_SVMCK2);

% [res.ave_TPR_SVMCK1 res.TPR_SVMCK1_std]=mean_stds_1(TPR_SVMCK1);
% [res.ave_TPR_SVMCK2 res.TPR_SVMCK2_std]=mean_stds_1(TPR_SVMCK2);

% [res.ave_AA_SVMCK1	res.AA_SVMCK1_std]=mean_stds_2(TPR_SVMCK1);
% [res.ave_AA_SVMCK2	res.AA_SVMCK2_std]=mean_stds_2(TPR_SVMCK2);
time_end3=toc(time_start3);
time_result.t1=time_end1;
time_result.t2=time_end2;
time_result.t3=time_end3;
end
function [mean_val,std_val]= mean_std(Values)
	mean_val=mean(Values);
	std_val=std(Values);
end
function [mean_val,std_val]= mean_stds_1(Values)
	mean_val=mean(Values,1);
	std_val=std(Values,1);
end
function [mean_val,std_val]= mean_stds_2(Values)
	mean_val=mean(mean(Values,2));
	std_val=std(mean(Values,2));
end