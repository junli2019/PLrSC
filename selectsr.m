function [train_d, test_d, train_t, test_t]=selectsr(featureMat,labelMat,selectnum,style)

% style==0 randm select the selectnum samples
if style==0
   Y=randperm(size(labelMat,2));
   Ytrain=sort(Y(1:selectnum));
   Ytest=sort(Y(selectnum+1:end));
   train_d=featureMat(:,Ytrain);
   test_d=featureMat(:,Ytest);
   train_t=labelMat(:,Ytrain);
   test_t=labelMat(:,Ytest);
end

% style==1  every class select the selectnum samples
if style==1
labelnum=max(labelMat);
num=[];
for ii=1:labelnum
    num=[num sum(labelMat==ii)];
end

train_d=[];
test_d=[];
train_t=[];
test_t=[];
allnum=0;
for i=1:labelnum
    feature=featureMat(:,allnum+1:allnum+num(i));
    label = labelMat(:,allnum+1:allnum+num(i));
    kk = randperm(num(i));
    train_d=[train_d feature(:,kk(1:selectnum))];
    train_t=[train_t label(:,kk(1:selectnum))];
    test_d=[test_d feature(:,kk(selectnum+1:num(i)))];
    test_t=[test_t label(:,kk(selectnum+1:num(i)))];
    allnum=allnum+num(i);
end
end

end