%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLrSC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jun Li, Hongfu Liu, Handong Zhao and Yun Fu, Projective Low-rank Subspace 
% Clustering via Learning Deep Encoder, IJCAI 2017.

clear all;
addpath(genpath('utility'));
load mnistsc2000;

opts.lambda=0.1;          % reguleration parameter for sparse noise
opts.epsilon = 0.0001;    % learning rate
opts.alpha= 0.00001;      % reguleration parameter for the weight of deep encoder
opts.act_fun = 'max';     % activation functions: 'sigm', 'tanh', 'max',
                          % 'shrink', 'softplus','linear'.
opts.hidnum= [1500 1500]; % structure of deep encoder;
opts.NNmaxiter=3;         % maxiter for training deep encoder.
opts.selectnum=50;        % the selected number of each cluster
opts.style = 1;           % 0: random selection from the big dataset;
                          % 1: random selection in each cluster from the big dataset;

tic;
[Data.train, Data.test, Data.train_label, Data.test_label]=...
    selectsr(DATA,labels,opts.selectnum,opts.style); % select small dataset
A=Data.train;
[acc,nmi,err]=solve_PLrSC(Data.train,Data.train_label,DATA,labels,A,opts);
alltime=toc;
disp(['all times:   ' num2str(alltime)]);

  