function r = fgm(source, target, name)
% clear variables;

%% src parameter
prSet(1);

%% src parameter
tag = 1;
nIn = 10; % #inliers
nOuts = [0 0]; % #outliers
egDen = .5; % edge density
egDef = 0; % edge deformation
parKnl = st('alg', 'toy'); % type of affinity: synthetic data

%% algorithm parameter
[pars, algs] = gmPar(2);

%% src
wsSrc = toyAsgSrcD(tag, nIn, nOuts, egDen, egDef);
[gphs, asgT] = stFld(wsSrc, 'gphs', 'asgT');


json{1} = jsondecode(source);
json{2} = jsondecode(target);

gphss{1} = rdGphU(json{1});
gphss{2} = rdGphU(json{2});


%% affinity
[KP, KQ] = conKnlGphPQD(gphs, parKnl); % node and edge affinity
K = conKnlGphKD(KP, KQ, gphs); % global affinity
Ct = ones(size(KP)); % mapping constraint (default to a constant matrix of one)

%% directed graph -> undirected graph (for fgmU and PM)
gphUs = gphD2Us(gphs);
[~, KQU] = knlGphKD2U(KP, KQ, gphUs);

%% Truth
asgT.obj = asgT.X(:)' * K * asgT.X(:);
asgT.acc = 1;

asgGa = gm(K, Ct, asgT, pars{1}{:});asgPm = pm(K, KQU, gphUs, asgT);asgSm = gm(K, Ct, asgT, pars{3}{:});asgSmac = gm(K, Ct, asgT, pars{4}{:});asgIpfpU = gm(K, Ct, asgT, pars{5}{:});asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
asgFgmU = fgmU(KP, KQU, Ct, gphUs, asgT, pars{8}{:});
asgFgmD = fgmD(KP, KQ, Ct, gphs, asgT, pars{9}{:});

r.Ga = asgGa.X;
r.Pm = asgPm.X;
r.Sm = asgSm.X;
r.Smac = asgSmac.X;
r.IpfpU = asgIpfpU.X;
r.IpfpS = asgIpfpS.X;
r.Rrwm = asgRrwm.X;
r.FgmU = asgFgmU.X;

r.sourceindex2id = gphs{1}.index2id
r.targetindex2id = gphs{2}.index2id