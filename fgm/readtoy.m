function r = readtoy()
% A demo comparison of different graph matching methods on the synthetic dataset.
%
% Remark
%   The edge is directed and the edge feature is asymmetric.
%
% History
%   create  -  Feng Zhou (zhfe99@gmail.com), 01-20-2012
%   modify  -  Feng Zhou (zhfe99@gmail.com), 05-07-2013

clear variables;
prSet(1);

%% src parameter
tag = 1;
nIn = 10; % #inliers
nOuts = [0 0]; % #outliers
egDen = 5; % edge density
egDef = 0; % edge deformation
% parKnl = st('alg', 'cmum'); % type of affinity: only edge distance
parKnl = st('alg', 'toy'); % type of affinity: synthetic data

%% algorithm parameter
[pars, algs] = gmPar(2);

%% src
wsSrc = toyAsgSrcD(tag, nIn, nOuts, egDen, egDef);
[gphs, asgT] = stFld(wsSrc, 'gphs', 'asgT');

gphs = gphD2Us(gphs);

r.pt1 = gphs{1}.Pt
r.eg1 = gphs{1}.Eg
r.pt2 = gphs{2}.Pt
r.eg2 = gphs{2}.Eg
r.grt = asgT.X
