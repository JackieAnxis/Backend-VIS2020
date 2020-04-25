function gph = rdGphU(val)
% Generate a graph by connecting points.
%
% Remark
%   The edge is directed and the edge feature is asymmetric.
%
% Input
%   path       -  graph node, d x n
%            
% Output     
%   gph      -  graph
%     Pt     -  graph node, d x n
%     Eg     -  graph edge, 2 x 2m
%     vis    -  binary indicator of nodes that have been kept, 1 x n | []
%     G      -  node-edge adjacency, n x m
%     H      -  augumented node-edge adjacency, n x (m + n)
%     PtD    -  edge feature, 2 x 2m
%     dsts   -  distance, 1 x 2m
%     angs   -  angle, 1 x 2m
%     angAs  -  angle, 1 x 2m
%            
% History    
%   create   -  Feng Zhou (zhfe99@gmail.com), 08-11-2011
%   modify   -  Feng Zhou (zhfe99@gmail.com), 05-07-2013

% dimension

%fileread(json)
% edge
% [Eg, vis] = gphEg(Pt, parGph);

Ptx = [];
Pty = [];
index2id = [];
n = size(val.nodes, 1);
for i=1:n
    node=val.nodes(i:i);
    x = node.x;
    y = node.y;
%    fprintf("%f, %f", x, y);
    Ptx = [Ptx, x];
    Pty = [Pty, y];
    id = str2num(node.id);
    % id = node.id;
    index2id = [index2id, id];
    fprintf('%d ', id);
end
Pt = [Ptx;Pty];
fprintf('\n');
Eg1 = [];
Eg2 = [];
m = size(val.links, 1);
for i=1:m
    link=val.links(i:i);
    source = str2num(link.source);
    target = str2num(link.target);
    source = find(index2id==source);
    target = find(index2id==target);
    Eg1 = [Eg1, source];
    Eg2 = [Eg2, target];
end
Eg = [[Eg1, Eg2];[Eg2, Eg1]];
%parG = st('link', 'del');
%[Eg, vis] = gphEg(Pt, parG);

%fprintf('Given links: ');
%for i=1:size(Eg1, 1)
%    fprintf('%d %d, ', Eg1(i:i), Eg2(i:i));
%end
%
%fprintf('\n');
%fprintf('Built links: ');
%for i=1:size(Eg, 1)
%    fprintf('%d %d, ', Eg(i:i));
%end

[G, H] = gphEg2IncU(Eg, n);

% second-order feature
[PtD, dsts, angs, angAs] = gphEg2Feat(Pt, Eg);

% store
% gph.vis = vis;
gph.Pt = Pt;
gph.Eg = Eg;
gph.vis = [];
gph.index2id = index2id;
gph.G = G;
gph.H = H;
gph.PtD = PtD;
gph.dsts = dsts;
gph.angs = angs;
gph.angAs = angAs;
