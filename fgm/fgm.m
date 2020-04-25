function r = fgm(source, target, name)
% clear variables;

%source = '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"x": 318.983870967742, "y": 162.887096774194, "id": "1"}, {"x": 244.661290322581, "y": 143.145161290323, "id": "4"}, {"x": 129.112903225806, "y": 157.08064516129, "id": "3"}, {"x": 371.241935483871, "y": 355.08064516129, "id": "0"}, {"x": 396.209677419355, "y": 180.887096774194, "id": "5"}, {"x": 342.209677419355, "y": 236.629032258065, "id": "2"}, {"x": 390.983870967742, "y": 180.887096774194, "id": "7"}, {"x": 84.4032258064516, "y": 261.016129032258, "id": "8"}, {"x": 282.403225806452, "y": 357.403225806452, "id": "9"}, {"x": 145.951612903226, "y": 327.209677419355, "id": "6"}, {"x": 116.91935483871, "y": 321.403225806452, "id": "10"}, {"x": 306.209677419355, "y": 136.177419354839, "id": "11"}, {"x": 291.693548387097, "y": 313.274193548387, "id": "12"}, {"x": 337.564516129032, "y": 187.854838709677, "id": "13"}, {"x": 355.564516129032, "y": 185.532258064516, "id": "14"}, {"x": 328.854838709677, "y": 287.725806451613, "id": "15"}, {"x": 239.435483870968, "y": 248.822580645161, "id": "16"}, {"x": 316.661290322581, "y": 259.274193548387, "id": "17"}, {"x": 340.467741935484, "y": 257.532258064516, "id": "18"}, {"x": 275.435483870968, "y": 339.403225806452, "id": "19"}, {"x": 327.112903225806, "y": 311.532258064516, "id": "21"}, {"x": 317.822580645161, "y": 238.951612903226, "id": "22"}, {"x": 166.854838709677, "y": 237.209677419355, "id": "23"}, {"x": 359.629032258065, "y": 75.7903225806451, "id": "24"}, {"x": 377.048387096774, "y": 78.6935483870967, "id": "20"}, {"x": 421.177419354839, "y": 144.306451612903, "id": "25"}, {"x": 290.532258064516, "y": 292.951612903226, "id": "26"}, {"x": 390.983870967742, "y": 159.403225806452, "id": "27"}], "links": [{"source": "1", "target": "4"}, {"source": "1", "target": "11"}, {"source": "1", "target": "13"}, {"source": "1", "target": "14"}, {"source": "1", "target": "16"}, {"source": "1", "target": "22"}, {"source": "1", "target": "27"}, {"source": "4", "target": "3"}, {"source": "4", "target": "11"}, {"source": "4", "target": "16"}, {"source": "4", "target": "23"}, {"source": "4", "target": "24"}, {"source": "3", "target": "8"}, {"source": "3", "target": "23"}, {"source": "3", "target": "24"}, {"source": "0", "target": "5"}, {"source": "0", "target": "9"}, {"source": "0", "target": "15"}, {"source": "0", "target": "18"}, {"source": "0", "target": "21"}, {"source": "0", "target": "25"}, {"source": "5", "target": "2"}, {"source": "5", "target": "7"}, {"source": "5", "target": "18"}, {"source": "5", "target": "25"}, {"source": "5", "target": "27"}, {"source": "2", "target": "7"}, {"source": "2", "target": "13"}, {"source": "2", "target": "14"}, {"source": "2", "target": "18"}, {"source": "2", "target": "22"}, {"source": "7", "target": "14"}, {"source": "7", "target": "27"}, {"source": "8", "target": "10"}, {"source": "8", "target": "23"}, {"source": "9", "target": "6"}, {"source": "9", "target": "10"}, {"source": "9", "target": "12"}, {"source": "9", "target": "19"}, {"source": "9", "target": "21"}, {"source": "6", "target": "10"}, {"source": "6", "target": "16"}, {"source": "6", "target": "19"}, {"source": "6", "target": "23"}, {"source": "10", "target": "23"}, {"source": "11", "target": "24"}, {"source": "11", "target": "27"}, {"source": "12", "target": "19"}, {"source": "12", "target": "21"}, {"source": "12", "target": "26"}, {"source": "13", "target": "14"}, {"source": "13", "target": "22"}, {"source": "14", "target": "27"}, {"source": "15", "target": "17"}, {"source": "15", "target": "18"}, {"source": "15", "target": "21"}, {"source": "15", "target": "26"}, {"source": "16", "target": "17"}, {"source": "16", "target": "19"}, {"source": "16", "target": "22"}, {"source": "16", "target": "23"}, {"source": "16", "target": "26"}, {"source": "17", "target": "18"}, {"source": "17", "target": "22"}, {"source": "17", "target": "26"}, {"source": "18", "target": "22"}, {"source": "19", "target": "26"}, {"source": "21", "target": "26"}, {"source": "24", "target": "20"}, {"source": "24", "target": "27"}, {"source": "20", "target": "25"}, {"source": "20", "target": "27"}, {"source": "25", "target": "27"}]}'
%target = '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"x": 168.016129032258, "y": 132.112903225806, "id": "1"}, {"x": 154.08064516129, "y": 257.532258064516, "id": "2"}, {"x": 378.209677419355, "y": 223.274193548387, "id": "0"}, {"x": 373.564516129032, "y": 189.596774193548, "id": "4"}, {"x": 393.306451612903, "y": 102.5, "id": "3"}, {"x": 325.951612903226, "y": 264.5, "id": "5"}, {"x": 331.177419354839, "y": 12.4999999999999, "id": "6"}, {"x": 314.338709677419, "y": 122.822580645161, "id": "8"}, {"x": 330.016129032258, "y": 150.112903225806, "id": "9"}, {"x": 208.08064516129, "y": 227.338709677419, "id": "10"}, {"x": 451.370967741936, "y": 254.629032258065, "id": "11"}, {"x": 415.370967741936, "y": 49.6612903225806, "id": "7"}, {"x": 368.338709677419, "y": 173.338709677419, "id": "12"}, {"x": 326.532258064516, "y": 315.016129032258, "id": "13"}, {"x": 207.5, "y": 303.403225806452, "id": "14"}, {"x": 332.338709677419, "y": 198.887096774194, "id": "15"}, {"x": 257.435483870968, "y": 96.1129032258064, "id": "16"}, {"x": 403.177419354839, "y": 94.9516129032257, "id": "17"}, {"x": 264.403225806452, "y": 234.887096774194, "id": "18"}, {"x": 385.177419354839, "y": 240.693548387097, "id": "19"}, {"x": 228.403225806452, "y": 306.306451612903, "id": "20"}, {"x": 352.661290322581, "y": 132.112903225806, "id": "21"}, {"x": 342.790322580645, "y": 12.4999999999999, "id": "22"}, {"x": 208.661290322581, "y": 341.145161290323, "id": "23"}, {"x": 345.693548387097, "y": 321.983870967742, "id": "24"}, {"x": 303.306451612903, "y": 83.9193548387096, "id": "25"}, {"x": 395.629032258065, "y": 76.9516129032257, "id": "26"}, {"x": 385.177419354839, "y": 84.5, "id": "27"}], "links": [{"source": "1", "target": "2"}, {"source": "1", "target": "6"}, {"source": "1", "target": "10"}, {"source": "1", "target": "16"}, {"source": "2", "target": "10"}, {"source": "2", "target": "14"}, {"source": "2", "target": "23"}, {"source": "0", "target": "4"}, {"source": "0", "target": "5"}, {"source": "0", "target": "11"}, {"source": "0", "target": "15"}, {"source": "0", "target": "19"}, {"source": "4", "target": "3"}, {"source": "4", "target": "11"}, {"source": "4", "target": "12"}, {"source": "4", "target": "15"}, {"source": "4", "target": "17"}, {"source": "3", "target": "12"}, {"source": "3", "target": "17"}, {"source": "3", "target": "21"}, {"source": "3", "target": "27"}, {"source": "5", "target": "13"}, {"source": "5", "target": "15"}, {"source": "5", "target": "18"}, {"source": "5", "target": "19"}, {"source": "5", "target": "20"}, {"source": "5", "target": "24"}, {"source": "6", "target": "16"}, {"source": "6", "target": "22"}, {"source": "6", "target": "25"}, {"source": "8", "target": "9"}, {"source": "8", "target": "16"}, {"source": "8", "target": "18"}, {"source": "8", "target": "21"}, {"source": "8", "target": "25"}, {"source": "9", "target": "12"}, {"source": "9", "target": "15"}, {"source": "9", "target": "18"}, {"source": "9", "target": "21"}, {"source": "10", "target": "14"}, {"source": "10", "target": "16"}, {"source": "10", "target": "18"}, {"source": "10", "target": "20"}, {"source": "11", "target": "7"}, {"source": "11", "target": "17"}, {"source": "11", "target": "19"}, {"source": "11", "target": "24"}, {"source": "7", "target": "17"}, {"source": "7", "target": "22"}, {"source": "7", "target": "26"}, {"source": "12", "target": "15"}, {"source": "12", "target": "21"}, {"source": "13", "target": "20"}, {"source": "13", "target": "23"}, {"source": "13", "target": "24"}, {"source": "14", "target": "20"}, {"source": "14", "target": "23"}, {"source": "15", "target": "18"}, {"source": "16", "target": "18"}, {"source": "16", "target": "25"}, {"source": "17", "target": "26"}, {"source": "17", "target": "27"}, {"source": "18", "target": "20"}, {"source": "19", "target": "24"}, {"source": "20", "target": "23"}, {"source": "21", "target": "25"}, {"source": "21", "target": "27"}, {"source": "22", "target": "25"}, {"source": "22", "target": "26"}, {"source": "22", "target": "27"}, {"source": "23", "target": "24"}, {"source": "25", "target": "27"}, {"source": "26", "target": "27"}]}'

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


%if strcmp(name, 'Ga')
    %% GA
    asgGa = gm(K, Ct, asgT, pars{1}{:});
    r.Ga = asgGa.X;
    % asgGaX = asgGa.X
    % save './data/test/ga.mat' asgGaX
%elseif strcmp(name, 'Pm')
    %% PM
    asgPm = pm(K, KQ, gphs, asgT);
    r.Pm = asgPm.X;
    % asgPmX = asgPm.X
    % save './data/test/pm.mat' asgPmX
%elseif strcmp(name, 'Sm')
    %% SM
    asgSm = gm(K, Ct, asgT, pars{3}{:});
    r.Sm = asgSm.X;
    % asgPmX = asgPm.X
    % save './data/test/sm.mat' asgPmX
%elseif strcmp(name, 'Smac')
    %% SMAC
    asgSmac = gm(K, Ct, asgT, pars{4}{:});
    r.Smac = asgSmac.X;
    % asgSmacX = asgSmac.X
    % save './data/test/smac.mat' asgSmacX
%elseif strcmp(name, 'IpfpU')
    %% IPFP-U
    asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
    r.IpfpU = asgIpfpU.X;
    % asgIpfpUX = asgIpfpU.X
% save './data/test/ipfpu.mat' asgIpfpUX
%elseif strcmp(name, 'IpfpS')
    %% IPFP-S
    asgIpfpS = gm(K, Ct, asgT, pars{6}{:});
    r.IpfpS = asgIpfpS.X;
    % asgIpfpSX = asgIpfpS.X
    % save './data/test/ipfps.mat' asgIpfpSX
%elseif strcmp(name, 'Rrwm')
    %% RRWM
    asgRrwm = gm(K, Ct, asgT, pars{7}{:});
    r.Rrwm = asgRrwm.X;
    % asgRrwmX = asgRrwm.X
    % save './data/test/rrwm.mat' asgRrwmX
%elseif strcmp(name, 'FgmD')
    %% FGM-D
%    asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
%    r.FgmD = asgFgmD.X;
    % asgFgmDX = asgFgmD.X
    % save './data/test/fgmd.mat' asgFgmDX
%elseif strcmp(name, 'FgmU)
    %% FGM-U
    asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
    r.FgmU = asgFgmU.X;
    % asgFgmUX = asgFgmU.X
    % save './data/test/fgmu.mat' asgFgmUX
%end

% %% print information
% fprintf('GA    : acc %.2f, obj %.2f\n', asgGa.acc, asgGa.obj);
% fprintf('PM    : acc %.2f, obj %.2f\n', asgPm.acc, asgPm.obj);
% fprintf('SM    : acc %.2f, obj %.2f\n', asgSm.acc, asgSm.obj);
% fprintf('SMAC  : acc %.2f, obj %.2f\n', asgSmac.acc, asgSmac.obj);
% fprintf('IPFP-U: acc %.2f, obj %.2f\n', asgIpfpU.acc, asgIpfpU.obj);
% fprintf('IPFP-S: acc %.2f, obj %.2f\n', asgIpfpS.acc, asgIpfpS.obj);
% fprintf('RRWM  : acc %.2f, obj %.2f\n', asgRrwm.acc, asgRrwm.obj);
% fprintf('FGM-U : acc %.2f, obj %.2f\n', asgFgmU.acc, asgFgmU.obj);
% fprintf('FGM-D : acc %.2f, obj %.2f\n', asgFgmD.acc, asgFgmD.obj);

% r.asgT = asgT.X
r.sourceindex2id = gphs{1}.index2id
r.targetindex2id = gphs{2}.index2id