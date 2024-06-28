function [Train_Test]=load_loc_train_test_ratio(img_name,i,ratio)

path=['./train_indexes2/' img_name '/'];
file_name=[img_name '_train_test' num2str(i) '_' num2str(ratio) '.mat'];
 load([path file_name]);
Train_Test.loc_train=loc_train;
Train_Test.loc_test=loc_test;
Train_Test.CTest=CTest;
Train_Test.CTrain=CTrain;
end

