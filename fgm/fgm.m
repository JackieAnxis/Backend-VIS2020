function r = fgm(sourcePath, targetPath)
% clear variables;

%% src parameter
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
asgT = st('alg', 'truth');

%% algorithm parameter
[pars, algs] = gmPar(2);

json{1} = jsondecode(fileread(sourcePath));
json{2} = jsondecode(fileread(targetPath));

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

%% GA
asgGa = gm(K, Ct, asgT, pars{1}{:});
% asgGaX = asgGa.X
% save './data/test/ga.mat' asgGaX

%% PM
asgPm = pm(K, KQ, gphs, asgT);
% asgPmX = asgPm.X
% save './data/test/pm.mat' asgPmX

%% SM
asgSm = gm(K, Ct, asgT, pars{3}{:});
% asgPmX = asgPm.X
% save './data/test/sm.mat' asgPmX

%% SMAC
asgSmac = gm(K, Ct, asgT, pars{4}{:});
% asgSmacX = asgSmac.X
% save './data/test/smac.mat' asgSmacX

%% IPFP-U
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
% asgIpfpUX = asgIpfpU.X
% save './data/test/ipfpu.mat' asgIpfpUX

%% IPFP-S
asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
% asgIpfpSX = asgIpfpS.X
% save './data/test/ipfps.mat' asgIpfpSX

%% RRWM
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
% asgRrwmX = asgRrwm.X
% save './data/test/rrwm.mat' asgRrwmX

%% FGM-U
asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
% asgFgmUX = asgFgmU.X
% save './data/test/fgmu.mat' asgFgmUX

%% FGM-D
asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
% asgFgmDX = asgFgmD.X
% save './data/test/fgmd.mat' asgFgmDX

%% GA
r.asgGa = asgGa.X
%% PM
r.asgPm = asgPm.X
%% SM
r.asgSm = asgSm.X
%% SMAC
r.asgSmac = asgSmac.X
%% IPFP-U
r.asgIpfpU = asgIpfpU.X
%% IPFP-S
r.asgIpfpS = asgIpfpS.X
%% RRWM
r.asgRrwm = asgRrwm.X
%% FGM-U
r.asgFgmU = asgFgmU.X
%% FGM-D
r.asgFgmD = asgFgmD.X

r.sourceindex2id = gphs{1}.index2id
% save './data/test/sourceindex2id.mat' sourceindex2id
r.targetindex2id = gphs{2}.index2id
% save './data/test/targetindex2id.mat' targetindex2id