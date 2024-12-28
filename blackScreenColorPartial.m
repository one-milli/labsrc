%% Simulate the capture of the patterns and their use for binary search
close all;
clc;
imaqreset

expDate = '241226';
trimRowFrom = 381;
trimRowTo = 930;
trimColFrom = 251;
trimColTo = 800;
m = 500;
n = 256;
nn = n * n;

if not(exist(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_R_', expDate], 'dir'))
    mkdir(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_R_', expDate]);
end

if not(exist(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_G_', expDate], 'dir'))
    mkdir(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_G_', expDate]);
end

if not(exist(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_B_', expDate], 'dir'))
    mkdir(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_B_', expDate]);
end

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
vid.FramesPerTrigger = 5;
triggerconfig(vid, 'manual');
vid.TriggerRepeat = Inf;
start(vid);

%% capture
% data = load('use_list256_5.0.mat');
data = load('use_list_manual.mat');
sta = 1;
fin = floor(nn * 0.05);

for k = sta:fin
    ind = data.use_list(k);
    image_disp = uint8(imread(['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_input/hadamard_', int2str(ind), '.png']));

    for channel = 1:3
        Line = zeros(wy_pro, wx_pro, 3, 'uint8');

        for i = 1:n

            for j = 1:n
                temp = zeros(mg, mg);
                temp(:) = image_disp(j, i);
                Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg, channel) = temp;
            end

        end

        disp(['i = ', int2str(k)])
        imshow(Line, 'Parent', ha);
        pause(1)

        trigger(vid);
        frames = getdata(vid, 5);
        avgImg = uint8(mean(frames, 4));
        img = avgImg(trimRowFrom:trimRowTo, trimColFrom:trimColTo, :);
        % img = imresize(img, [m m]);

        if channel == 1
            imwrite(img, ['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_R_', expDate, '/hadamard_' int2str(ind), '.png'], 'BitDepth', 8);
        elseif channel == 2
            imwrite(img, ['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_G_', expDate, '/hadamard_' int2str(ind), '.png'], 'BitDepth', 8);
        elseif channel == 3
            imwrite(img, ['../../OneDrive - m.titech.ac.jp/Lab/data/hadamard', int2str(n), '_cap_B_', expDate, '/hadamard_' int2str(ind), '.png'], 'BitDepth', 8);
        end

    end

end

%% Stop camera
stop(vid);
delete(vid);
clear vid
