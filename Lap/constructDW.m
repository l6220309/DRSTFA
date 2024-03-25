function W= constructDW(X,options)
dist=EuDist2(X);
[r,l]=size(X);
G=zeros(r,r);
label=options.label;
for i=1:r
    for j=1:r
        if label(i,1)==label(j,1)
            dist(i,j)=inf;
        end
    end
end
[newdist1,loc1]=sort(dist,2);
loc1=loc1(:,1:options.k);
[row1,col1]=size(loc1);
for i=1:row1
        for j=1:col1
            G(i,loc1(i,j))=1;
        end
end
    w1=G;
    for i=1:r
        for j=1:r
            if (G(i,j)==0)&&(w1(j,i)==1)
                G(i,j)=1;
            end
        end
    end
      W= sparse(G);
%W=constructW(X,options);