function r = fgm(source, target, name)
% clear variables;

%% src parameter
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
asgT = st('alg', 'truth');

%% algorithm parameter
[pars, algs] = gmPar(2);

%json{1} = jsondecode(fileread(source));
%json{2} = jsondecode(fileread(target));
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
    asgFgmD = fgmD(KP, KQD, Ct, gphDs, asgT, pars{9}{:});
    r.FgmD = asgFgmD.X;
    % asgFgmDX = asgFgmD.X
    % save './data/test/fgmd.mat' asgFgmDX
%elseif strcmp(name, 'FgmU)
    %% FGM-U
    asgFgmU = fgmU(KP, KQ, Ct, gphs, asgT, pars{8}{:});
    r.FgmU = asgFgmU.X;
    % asgFgmUX = asgFgmU.X
    % save './data/test/fgmu.mat' asgFgmUX
%end

r.sourceindex2id = gphs{1}.index2id
% save './data/test/sourceindex2id.mat' sourceindex2id
r.targetindex2id = gphs{2}.index2id
% save './data/test/targetindex2id.mat' targetindex2id