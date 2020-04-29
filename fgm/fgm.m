function r = fgm(source, target, name)
% clear variables;

%% src parameter
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
asgT = st('alg', 'truth');

%% algorithm parameter
[pars, algs] = gmPar(2);

json{1} = jsondecode(source);
json{2} = jsondecode(target);

gphs{1} = rdGphU(json{1});
gphs{2} = rdGphU(json{2});

asgT.X = [];

%% affinity
[KP, KQ] = conKnlGphPQU(gphs, parKnl);
K = conKnlGphKU(KP, KQ, gphs);
Ct = ones(size(KP));

%% undirected graph -> directed graph (for FGM-D)
gphDs = gphU2Ds(gphs);
KQD = [KQ, KQ; KQ, KQ];

% prSet(1);
% %% src parameter
% test{1} = '1'
% test{2} = '2'
% tag = 'house';
% pFs = [1 100]; % frame index
% nIn = [30 30] - 2; % randomly remove 2 nodes
% parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
%
% %% algorithm parameter
% [pars, algs] = gmPar(2);
%
% %% src
% wsSrc = cmumAsgSrc(tag, pFs, nIn, 'svL', 1);
% asgT = wsSrc.asgT;
%
% %% feature
% parG = st('link', 'del'); % Delaunay triangulation for computing the graphs
% parF = st('smp', 'n', 'nBinT', 4, 'nBinR', 3); % not used, ignore it
% wsFeat = cmumAsgFeat(wsSrc, parG, parF, 'svL', 1);
% [gphs, XPs, Fs] = stFld(wsFeat, 'gphs', 'XPs', 'Fs');

%% affinity
% [KP, KQ] = conKnlGphPQU(gphs, parKnl);
% K = conKnlGphKU(KP, KQ, gphs);
% Ct = ones(size(KP));

%%% undirected graph -> directed graph (for FGM-D)
%gphDs = gphU2Ds(gphs);
%KQD = [KQ, KQ; KQ, KQ];



%% GA
asgGa = gm(K, Ct, asgT, pars{1}{:});
r.Ga = asgGa.X;

%% PM
asgPm = pm(K, KQ, gphs, asgT);
r.Pm = asgPm.X;

%% SM
asgSm = gm(K, Ct, asgT, pars{3}{:});
r.Sm = asgSm.X;

%% SMAC
asgSmac = gm(K, Ct, asgT, pars{4}{:});
r.Smac = asgSmac.X;

%% IPFP-U
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
r.IpfpU = asgIpfpU.X;

%% IPFP-S
asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
r.IpfpS = asgIpfpS.X;

%% RRWM
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
r.Rrwm = asgRrwm.X;

%% FGM-D
%asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
%r.FgmD = asgFgmD.X;

%% FGM-U
asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
r.FgmU = asgFgmU.X;

r.sourceindex2id = gphs{1}.index2id
r.targetindex2id = gphs{2}.index2id