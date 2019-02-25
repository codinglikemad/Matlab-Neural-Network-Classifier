close all
clear all
%%%% Example code for matlab neural network classifier %%%%
% Author: CodingLikeMad Youtube Channel, Febuary 2019


%% Data Loading

% Load the dataset to a table
tbl = readtable('shortColorDataset.csv');

% Print all unique colornames
unique(tbl.colorname)

% Translate the unique color names into a numeric code
[G,ID] = findgroups(tbl.colorname);

%% Data processining
% Build a one hot encoding using the grouped color data
target = ind2vec((G).');

% Build a 3xN feature matrix for training/testing of the neural network
x = [tbl.r.';tbl.g.';tbl.b.'];

%% Training and testing the network

% Make the neural net - patternnet does most of the work for us here
net = patternnet(10);

% Take the network and return the trained network using x and target for
% training. Note that matlab will INSIDE the function do test, train and
% validation splits. If you wanted a FAIR comparison, you would only study
% the TEST data set (or even better - do it yourself!). I don't do that
% here since I want to see what the neural net thinks and this dataset is
% very dense.
net = train(net,x,target);

% Nice little function to pop up a network structure visual.
view(net)

% This is how you APPLY the data. Note I am applying it on the full
% dataset, irrespective of test/train status. Performance should ONLY be on
% test set, and train performance vs. test performance is a good indication
% of over training. I wish I said this when I recorded.
y = net(x);

% This is a way of checking performance, but honestly it's better to do it
% yourself. IT doesn't have the vast majority of what I want to know in
% terms of KPIs.
perf = perform(net,target,y);

% According to mathworks this requires a vector with a single 1 in it, but
% keep in mind that our output is dense! So why does this work? Turns out
% internally it reports the index of the max entry, NOT where the 1 is.
% It's probably not good to depend on this staying that way forever, but
% it's conveniently exactly the 'softmax' opperation we need here.
classes = vec2ind(y);

%% Plot the confusion matrix large enough to actually see it.
accum_corr = [];
for ii = 1:length(ID)
    for jj = 1:length(ID)
        accum_corr(ii,jj) = sum( (classes == ii) & (G.' == jj));
    end
end

figure;
imagesc(accum_corr);
xticks([1:24]);
xticklabels( ID )
xtickangle(45);
yticks([1:24]);
yticklabels( ID )
title('Color Confusion Matrix', 'fontSize', 14);
set(gca, 'fontsize', 14);
