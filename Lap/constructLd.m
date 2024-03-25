function Ld=constructLd(X,beta,label)
%fea=[feas';feat'];
%[row,col]=size(fea);
options= [];
options.NeighborMode = 'Supervised';
options.k =10;
options.WeightMode = 'Binary';
options.gnd=label;
Wc=constructW(X',options);
DColc = full(sum(Wc,2));
Dc = spdiags(DColc,0,speye(size(Wc,1)));


%options1 = [];
%options1.NeighborMode = 'Supervised';
options1.k =3;%3;%5;
options1.label=label;
Wp=constructDW(X',options1);
%Wts=constructComW(feat',feas',options1);
DColp = full(sum(Wp,2));
Dp= spdiags(DColp,0,speye(size(Wp,1)));
Ld =2*beta*(Dc - Wc)-2*(1-beta)*(Dp - Wp);
%Ld =2*(Dc - Wc)-2*beta*(Dp - Wp);