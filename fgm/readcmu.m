function r = readcmu(frame1, frame2, rmc)
% A demo comparison of different graph matching methods on the on CMU House dataset.
%
% Remark
%   The edge is directed and the edge feature is asymmetric.
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-07-2013

% clear variables;
prSet(1);

%% src parameter
test{1} = '1'
test{2} = '2'
tag = 'house';
% pFs = [1 100]; % frame index
% nIn = [30 30] - 2; % randomly remove 2 nodes
pFs = [frame1 frame2]; % frame index
nIn = [30 30] - 0; % randomly remove 2 nodes
parKnl = st('alg', 'cmum'); % type of affinity: only edge distance

%% algorithm parameter
[pars, algs] = gmPar(2);

%% src
wsSrc = cmumAsgSrc(tag, pFs, nIn, 'svL', 1);
asgT = wsSrc.asgT;

%% feature
parG = st('link', 'del'); % Delaunay triangulation for computing the graphs
parF = st('smp', 'n', 'nBinT', 4, 'nBinR', 3); % not used, ignore it
wsFeat = cmumAsgFeat(wsSrc, parG, parF, 'svL', 1);
[gphs, XPs, Fs] = stFld(wsFeat, 'gphs', 'XPs', 'Fs');

r.pt1 = gphs{1}.Pt
r.eg1 = gphs{1}.Eg
r.pt2 = gphs{2}.Pt
r.eg2 = gphs{2}.Eg
r.grt = asgT.X
