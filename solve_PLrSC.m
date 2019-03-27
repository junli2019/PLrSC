function [ACC,NMI,err]=solve_PLrSC(Y,train_labels,DATA,labels,A,opts)
% This routine solves the following PLrSC problem,
% min |Z|_*+lambda*|E|_L+gamma*|Z-f_de(Y)|_F^2
% s.t., Y = AZ+E
% inputs:
%        Y -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
%        A -- D*N data matrix, it is the (self-representive) dictionary, 
%        lambda, gamma -- parameters
% Note: For simplicity, we first learn the low-rank representations, and 
%       then train the deep encoder. Thus, gamma is usully set to 1.      
 
%% Initializing weights
d=size(Y,1);
m=size(A,2);
hidnum=[d opts.hidnum m];
for ii=2:size(hidnum,2)
Weight{ii}=0.01*randn(hidnum(ii),hidnum(ii-1));
end

%%
% learning low-rank representations using any lrr codes.
Q = orth(A');
B = A*Q;
[Z,E] = preinexact_alm_lrr_l21(Y,B,opts);
Z = Q*Z;
%%
disp(['training deep neural networks----- start ']);
[Weight,Zlearn,err]=Learnmap(Weight,Z,Y,opts);
disp(['training deep neural networks----- end ']);
tic;
H=actfun(Weight,DATA,opts.act_fun);
codingtime=toc;

[accplrtrain, nmiplrtrain]=accncutLSC(Zlearn',train_labels');
[accplrall, nmiplrall]=accncutLSC(H{end}',labels');

disp(['train acc:     ' num2str(accplrtrain)]);
disp(['train nmi:     ' num2str(nmiplrtrain)]);
disp(['all data acc:  ' num2str(accplrall)]);
disp(['all data nmi:  ' num2str(nmiplrall)]);
disp(['coding time:   ' num2str(codingtime)]);
ACC=[accplrtrain,accplrall];
NMI=[nmiplrtrain,nmiplrall];

end