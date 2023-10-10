%% Simulate the capture of the patterns and their use for binary search
imaqreset

config = Config(128, 64, '230922', 1e-2);
expDate = '230922';
trimRowFrom = 400;
trimRowTo = 850;
trimColFrom = 400;
trimColTo = 850;
n = 128;
nn = n * n;

pause('on')

%Projector resolution
wx_pro = 1920;
wy_pro = 1200;
%PC monitor resolution
wx_pc = 1366;
wy_pc = 768;
%Camera resolution
wx_cam = 1280;
wy_cam = 1024;

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

sta = 2;
fin = 2;

%% Load Hadamard
hadamard = zeros(n, n, nn);

for k = sta:fin
    input = uint8(imread(['../data/Hadamard', int2str(n), '_input/hadamard', int2str(k), '.png']));
    input = imresize(input, [n, n]);
    hadamard(:, :, k) = input;
end

%% Camera Settings
Cam_mode = 'F7_RGB_1280x1024_Mode0';
vid = videoinput('pointgrey', 1, Cam_mode);
src = getselectedsource(vid);
mg = 2; %magnification
ulc = 500;
ulr = 860;
vid.FramesPerTrigger = 1;
triggerconfig(vid, 'manual');
vid.TriggerRepeat = Inf;
start(vid);

%% capture white
if sta == 1
    Line = zeros(wy_pro, wx_pro);

    for i = 1:n

        for j = 1:n
            temp = zeros(mg, mg);
            temp(:) = hadamard(j, i, 1);
            Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg) = temp;
        end

    end

    disp('capture white')
    imshow(Line, 'Parent', ha);

    pause(2)

    trigger(vid);
    img = getdata(vid, 1);

    imwrite(img, ['../data/capture_', expDate, '/capturewhite.png'], 'BitDepth', 8);
end

%% capture
for k = sta:fin

    for col = 1:3

        Line = zeros(wy_pro, wx_pro, 3);

        for i = 1:n

            for j = 1:n
                temp = zeros(mg, mg);
                temp(:) = hadamard(j, i, k);
                Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg, col) = temp;
            end

        end

        if col == 1
            disp(['R ', int2str(k)])
        elseif col == 2
            disp(['G ', int2str(k)])
        else
            disp(['B ', int2str(k)])
        end

        imshow(Line, 'Parent', ha);

        if k == sta
            pause(2)
        else
            pause(0.5)
        end

        trigger(vid);
        img = getdata(vid, 1);
        img = img(trimRowFrom:trimRowTo, trimColFrom:trimColTo, :);

        if col == 1
            imwrite(img, ['../data/hadamard_cap_R_', expDate, '/hadamard', int2str(n), '_', int2str(k), '.png'], 'BitDepth', 8);
        elseif col == 2
            imwrite(img, ['../data/hadamard_cap_G_', expDate, '/hadamard', int2str(n), '_', int2str(k), '.png'], 'BitDepth', 8);
        else
            imwrite(img, ['../data/hadamard_cap_B_', expDate, '/hadamard', int2str(n), '_', int2str(k), '.png'], 'BitDepth', 8);
        end

    end

end

%% Stop camera
stop(vid);
delete(vid);
clear vid
