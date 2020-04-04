clear variables;
prSet(1);

%% src parameter
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
asgT = st('alg', 'truth');

%% algorithm parameter
[pars, algs] = gmPar(2);

gphs{1} = rdGphU('./data/test/source.json');
gphs{2} = rdGphU('./data/test/target.json');

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
asgGaX = asgGa.X
save './data/test/ga.mat' asgGaX

%% PM
asgPm = pm(K, KQ, gphs, asgT);
asgPmX = asgPm.X
save './data/test/pm.mat' asgPmX

%% SM
asgSm = gm(K, Ct, asgT, pars{3}{:});
asgPmX = asgPm.X
save './data/test/sm.mat' asgPmX

%% SMAC
asgSmac = gm(K, Ct, asgT, pars{4}{:});
asgSmacX = asgSmac.X
save './data/test/smac.mat' asgSmacX

%% IPFP-U
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
asgIpfpUX = asgIpfpU.X
save './data/test/ipfpu.mat' asgIpfpUX

%% IPFP-S
asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
asgIpfpSX = asgIpfpS.X
save './data/test/ipfps.mat' asgIpfpSX

%% RRWM
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
asgRrwmX = asgRrwm.X
save './data/test/rrwm.mat' asgRrwmX

%% FGM-U
asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
asgFgmUX = asgFgmU.X
save './data/test/fgmu.mat' asgFgmUX

%% FGM-D
asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
asgFgmDX = asgFgmD.X
save './data/test/fgmd.mat' asgFgmDX

sourceindex2id = gphs{1}.index2id
save './data/test/sourceindex2id.mat' sourceindex2id

targetindex2id = gphs{2}.index2id
save './data/test/targetindex2id.mat' targetindex2id