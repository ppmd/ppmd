data=dlmread('compare_1.txt');


p_1 = data(data(:,1)==1,:);
p_2 = data(data(:,1)==2,:);
p_4 = data(data(:,1)==4,:);


sizes = ([7,8,9,10,11,12].^3)*23;

x=[];
dlpoly1=[];
ppmd1=[];

dlpoly2=[];
ppmd2=[];


dlpoly4=[];
ppmd4=[];



for s = sizes
    x=[x,s];
    sdata = p_1(p_1(:,2)==s,3:5);
    
    p=polyfit(sdata(:,1),sdata(:,2),1)./s;
    dlpoly1=[dlpoly1, p(1)];
    
    p=polyfit(sdata(:,1),sdata(:,3),1)./s;
    
    ppmd1=[ppmd1, p(1)];
    

end



for s = sizes
    
    sdata = p_2(p_2(:,2)==s,3:5);
    
    p=polyfit(sdata(:,1),sdata(:,2),1)./s;
    dlpoly2=[dlpoly2, p(1)];
    
    p=polyfit(sdata(:,1),sdata(:,3),1)./s;
    
    ppmd2=[ppmd2, p(1)];
    

end

for s = sizes
    
    sdata = p_4(p_4(:,2)==s,3:5);
    
    p=polyfit(sdata(:,1),sdata(:,2),1)./s;
    dlpoly4=[dlpoly4, p(1)];
    
    p=polyfit(sdata(:,1),sdata(:,3),1)./s;
    
    ppmd4=[ppmd4, p(1)];
    

end






plot(x,dlpoly4,x,ppmd4)








