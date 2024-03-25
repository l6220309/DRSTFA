clc; clearvars; close all; warning off all;
rng('default');
%% Compare raw and aligned features on MI
%% Leave-one subject-out
%% need to enable covariancetoolbox
addpath('./Lap')
mAcc=cell(1,2); mTime=cell(1,2);

%% make data
dataFolder=['./data/'];
files=dir([dataFolder 'A*.mat']);
XRaw=[]; yAll=[]; XAlignE=[]; XAlignR=[]; XAlignLE=[];
for s=1:length(files)
    s
    load([dataFolder files(s).name]);
    XRaw=cat(3,XRaw,x);
    yAll=cat(1,yAll,y);
    nTrials=length(y);
    RtE=mean(covariances(x),3); % reference state, Euclidean space
    RtR=riemann_mean(covariances(x)); % reference state, Riemmanian space
    RtLE=logeuclid_mean(covariances(x)); 
    sqrtRtE=RtE^(-1/2); 
    sqrtRtR=RtR^(-1/2);
    sqrtRtLE=RtLE^(-1/2);
    XR=nan(size(x,1),size(x,2),nTrials);
    XE=nan(size(x,1),size(x,2),nTrials);
    XLE=nan(size(x,1),size(x,2),nTrials);
    for j=1:nTrials
        XR(:,:,j)=sqrtRtR*x(:,:,j);
        XE(:,:,j)=sqrtRtE*x(:,:,j);
        XLE(:,:,j)=sqrtRtLE*x(:,:,j);
    end
    XAlignE=cat(3,XAlignE,XE); 
    XAlignR=cat(3,XAlignR,XR);
    XAlignLE=cat(3,XAlignLE,XLE);
end
F=[];
tic;
Accs=cell(1,length(files));
times=cell(1,length(files));
for t=1:length(files)    %  target user
    t
    yt=yAll((t-1)*nTrials+1:t*nTrials);
    ys=yAll([1:(t-1)*nTrials t*nTrials+1:end]);
    XtRaw=XRaw(:,:,(t-1)*nTrials+1:t*nTrials);
    XsRaw=XRaw(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);
    XtAlignE=XAlignE(:,:,(t-1)*nTrials+1:t*nTrials);
    XsAlignE=XAlignE(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);
%         XtAlignR=XAlignR(:,:,(t-1)*nTrials+1:t*nTrials);
%         XsAlignR=XAlignR(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);
%         XtAlignLE=XAlignLE(:,:,(t-1)*nTrials+1:t*nTrials);
%         XsAlignLE=XAlignLE(:,:,[1:(t-1)*nTrials t*nTrials+1:end]);

%         %% mdRm
%         Cs_R = covariances(XsAlignR);
%         mean_Ct_R = riemann_mean(covariances(XtAlignR));
%         
%         Cs_LE = covariances(XsAlignLE);
%         mean_Ct_LE = logeuclid_mean(covariances(XtAlignLE));

%         Cs_E = covariances(XsAlignE);
%         mean_Ct_E = mean(covariances(XtAlignE),3);


%         mean_Ct = logeuclid_mean(covariances(XtAlignLE));
%         mean_Ct = mean(Ct,3);
%         ids = nTrials*(length(files)-1);
%         d = zeros(ids,1);
%         for k = 1:ids
%             d(k) =  distance_logeuclid(squeeze(Cs_E(:,:,k)),mean_Ct_E);
% %             d(k) =  distance_logeuclid(squeeze(Cs(:,:,k)),mean_Ct);
% %             d(k) =  distance(squeeze(Cs(:,:,k)),mean_Ct);
%         end

%         d_f = d(:,1);
%         [d_Y,d_I]=sort(d_f);
%         ids_i = d_I(1:500);
%         XsAlignE_i = XsAlignE(:,:,ids_i);
%         ys_i = ys(ids_i);

%         ys = ys_i;
%         XsAlignE = XsAlignE_i;


    %% CSP+LDA
    nFilter = 6;
    [Xs,Xt]=CSPfeature(XsAlignE,ys,XtAlignE,nFilter);         

    Xs = Xs';
    Xt = Xt';

    options.gamma = 1.0;
    options.kernel_type = 'rbf'; %2a rbf  %2b primal % 4a rbf
    options.lambda = 1.0;
    options.dim = 10;
    options.mu = 0.1;  % Target
    options.beta = 0.01; % Laplacian
    options.T = 10;

    L = constructL(Xs, Xt, 1);
    T = 5;
    Cls = []; Res = [];
    [ms,ns] = size(Xs);
    [mt,nt] = size(Xt);
    max_acc = 0;
    for m = 1:T
        [Zs,Zt] = Disciminant_JPDA(Xs,Xt,ys,Cls,L,options);  
        LDA = fitcdiscr(Zs',ys, 'discrimType','pseudoLinear');
        Cls = predict(LDA,Zt');
        acc = length(find(Cls==yt))/length(yt);
        if acc >= max_acc
            max_acc = acc;
        else
            break;
        end
    end
    Accs{t}(1)=max_acc*100;
    times{t}(1)=toc;
end
toc
ds = 1;
mAcc{ds}=[]; mTime{ds}=[];
for t=1:length(files)
    mAcc{ds}=cat(1,mAcc{ds},Accs{t});
    mTime{ds}=cat(1,mTime{ds},times{t});
end
mAcc{ds}=cat(1,mAcc{ds},mean(mAcc{ds}));
mTime{ds}=cat(1,mTime{ds},mean(mTime{ds}));
F = mAcc{ds}
FF = sqrt(var(F(1:end-1)))


save('MIall.mat','mAcc','mTime');