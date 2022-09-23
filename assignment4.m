%% clear workspace, remove figures and define variables
clc
clear
close all

addpath('./mnist');


%classes to test
classes = [1,8;3,8;1,7;5,6];



%select set size
setSize = 500;

%repeat for each pair of classes
for i=1:size(classes,1)
    
    %load set
    [setA,labelA] = loadMNIST(0,classes(i,1));
    [setB,labelB] = loadMNIST(0,classes(i,2));
    
    %build total set
    randIdxA = randperm(size(setA,1),floor(setSize/2));
    randIdxB = randperm(size(setB,1),ceil(setSize/2));
    set = [setA(randIdxA,:);setB(randIdxB,:);];
    %% train autoencoder
    
    myAutoencoder = trainAutoencoder(set',2);
    myEncodedData = encode(myAutoencoder,set');
    
    
    %% plot data
    figure
    plotcl(myEncodedData',[labelA(1:floor(setSize/2));labelB(1:ceil(setSize/2))]);
    legend(['Class ', num2str(labelA(1))], ['Class ', num2str(labelB(1))]);
    xlabel('Hidden unit 1');
    ylabel('Hidden unit 2');
    title(['Autoencoder classes ',num2str(classes(i,1)), ' and ', +num2str(classes(i,2))]);
    
end


