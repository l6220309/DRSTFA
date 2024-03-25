function [W1,W2] = constructComW(feas,feat,options)

if strcmpi(options.NeighborMode,'Supervised')
%fea=[feas;feat];
%[r,cl]=size(feas);
[nSmps,c]=size(feas);
[nSmpt,c]=size(feat);
labels=options.label(1:nSmps,1);
labelt=options.label(nSmps+1:end,1);
 G1 = zeros(nSmps,nSmpt);
                 for i=1:nSmps 
        for j=1:nSmpt
           if labels(i,1)==labelt(j,1)
                    G1(i,j) = 1;
           else
               G1(i,j)=0;
           end
        end
                 end   
       W1 = sparse(G1);
       G2 = zeros(nSmpt,nSmps);
                 for i=1:nSmpt 
        for j=1:nSmps
           if labelt(i,1)==labels(j,1)
                    G2(i,j) = 1;
           else
               G2(i,j)=0;
           end
        end
                 end   
       W2 = sparse(G2);
end

if strcmpi(options.NeighborMode,'KNN')
    [nSmps,c]=size(feas);
    [nSmpt,c]=size(feat);
    dist1=EuDist2(feas,feat);
    dist2=EuDist2(feat,feas);
    if ~isfield(options,'k')
            options.k = 2;
    end
    [newdist1,loc1]=sort(dist1,2);
    [newdist2,loc2]=sort(dist2,2);
    loc1=loc1(:,1:options.k);
    loc2=loc2(:,1:options.k);
    [row1,col1]=size(loc1);
    [row2,col2]=size(loc2);
     G1 = zeros(nSmps,nSmpt);
     G2 = zeros(nSmpt,nSmps);
    for i=1:row1
        for j=1:col1
            G1(i,loc1(i,j))=1;
        end
    end
    
     for i=1:row2
        for j=1:col2
            G2(i,loc2(i,j))=1;
        end
     end
    w1=G1;
    w2=G2;
    for i=1:nSmps
        for j=1:nSmpt
            if (G1(i,j)==0)&&(w2(j,i)==1)
                G1(i,j)=1;
            end
        end
    end
    for i=1:nSmpt
        for j=1:nSmps
            if (G2(i,j)==0)&&(w1(j,i)==1)
                G2(i,j)=1;
            end
        end
    end
    W1= sparse(G1);
    W2= sparse(G2);
end












