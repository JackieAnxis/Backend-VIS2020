clear variables;
prSet(1);

json{1} = fileread('../data/VIS/result/interpolation0.json');
json{2} = fileread('../data/VIS/result/target5.json');
r = fgm(json{1}, json{2}, 'FgmU');
%source = st('value', './data/test/source.json');
%target = st('value', './data/test/target.json');
%source = './data/test/source.json';
%target = './data/test/target.json';
%r = fgm(source, target)
r.x = 0;
%clear variables;
%prSet(1);
%
%%% src parameter
%parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
%asgT = st('alg', 'truth');
%
%%% algorithm parameter
%[pars, algs] = gmPar(2);
%
%json{1} = fileread('./data/test/source.json')
%json{2} = fileread('./data/test/target.json')
%
%gphs{1} = rdGphU(json{1});
%gphs{2} = rdGphU(json{2});
%
%asgT.X = [];
%
%%% affinity
%[KP, KQ] = conKnlGphPQU(gphs, parKnl);
%K = conKnlGphKU(KP, KQ, gphs);
%Ct = ones(size(KP));
%
%%% undirected graph -> directed graph (for FGM-D)
%gphDs = gphU2Ds(gphs);
%KQD = [KQ, KQ; KQ, KQ];
%
%%% GA
%asgGa = gm(K, Ct, asgT, pars{1}{:});
%asgGaX = asgGa.X
%save './data/test/ga.mat' asgGaX
%
%%% PM
%asgPm = pm(K, KQ, gphs, asgT);
%asgPmX = asgPm.X
%save './data/test/pm.mat' asgPmX
%
%%% SM
%asgSm = gm(K, Ct, asgT, pars{3}{:});
%asgPmX = asgPm.X
%save './data/test/sm.mat' asgPmX
%
%%% SMAC
%asgSmac = gm(K, Ct, asgT, pars{4}{:});
%asgSmacX = asgSmac.X
%save './data/test/smac.mat' asgSmacX
%
%%% IPFP-U
%asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
%asgIpfpUX = asgIpfpU.X
%save './data/test/ipfpu.mat' asgIpfpUX
%
%%% IPFP-S
%asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
%asgIpfpSX = asgIpfpS.X
%save './data/test/ipfps.mat' asgIpfpSX
%
%%% RRWM
%asgRrwm = gm(K, Ct, asgT, pars{7}{:});
%asgRrwmX = asgRrwm.X
%save './data/test/rrwm.mat' asgRrwmX
%
%%% FGM-U
%asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
%asgFgmUX = asgFgmU.X
%save './data/test/fgmu.mat' asgFgmUX
%
%%% FGM-D
%asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
%asgFgmDX = asgFgmD.X
%save './data/test/fgmd.mat' asgFgmDX
%
%sourceindex2id = gphs{1}.index2id
%save './data/test/sourceindex2id.mat' sourceindex2id
%
%targetindex2id = gphs{2}.index2id
%save './data/test/targetindex2id.mat' targetindex2id