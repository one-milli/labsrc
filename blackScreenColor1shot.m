%% Simulate the capture of the patterns and their use for binary search
close all;
clc;
imaqreset

expDate = '241226';
trimRowFrom = 381;
trimRowTo = 930;
trimColFrom = 301;
trimColTo = 850;
m = 500;
n = 256;
nn = n * n;

pause('on')

%Projector resolution
wx_pro = 1920;
wy_pro = 1200;
%PC monitor resolution
wx_pc = 1366;
wy_pc = 768;
%Camera resolution
% wx_cam = 1280;
% wy_cam = 1024;

Nnumx = round(log2(wx_pro)); %=11
Nnumy = round(log2(wy_pro)) + 1; %=10+1 to cover all the pixels

% Screen region definition
rect_pro = [wx_pc + 1 wy_pc - wy_pro + 1 wx_pro wy_pro]; %Origin(pc lower left)
rect_pc = [1 wy_pc wx_pc wy_pc];
rect_pc_confirm = [floor(wx_pc / 4) floor(wy_pc / 4) floor(wx_pc / 2) floor(wy_pc / 2)];

%% Figure initialization
set(0, 'defaultfigureposition', rect_pro);
h = figure('Units', 'pixels', 'DockControls', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
ha = axes('Parent', h, 'Units', 'pixels', 'Position', [1 1 wx_pro wy_pro]);

%% Camera Settings
vid = videoinput('pointgrey', 1, 'F7_RGB_1288x964_Mode0');
src = getselectedsource(vid);
mg = 1; %magnification
ulc = 500;
ulr = 750;
vid.FramesPerTrigger = 1;
triggerconfig(vid, 'manual');
vid.TriggerRepeat = 1;
start(vid);

%% capture
filename = 'Parrots';
image_disp = uint8(imread(['../../OneDrive - m.titech.ac.jp/Lab/data/sample_image_col', int2str(n), '/', filename, '.bmp']));
% input = imresize(input, [n, n]);
% image_disp = input;

Line = zeros(wy_pro, wx_pro, 3);

for i = 1:n

    for j = 1:n
        temp = zeros(mg, mg, 3);
        temp(:) = image_disp(j, i, :);
        Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg, :) = temp;
    end

end

disp('snapshot')
imshow(uint8(Line), 'Parent', ha);

pause(2)

trigger(vid);
img = getdata(vid, 1);
img = img(trimRowFrom:trimRowTo, trimColFrom:trimColTo, :);
% img = imresize(img, [m m]);

imwrite(img, ['../../OneDrive - m.titech.ac.jp/Lab/data/capture_', expDate, '/', filename, '.png'], 'BitDepth', 8);

%% Stop camera
stop(vid);
delete(vid);
clear vid