%% Simulate the capture of the patterns and their use for binary search
imaqreset

expDate = '240814';
trimRowFrom = 351;
trimRowTo = 850;
trimColFrom = 371;
trimColTo = 870;
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

%% Camera Settings
Cam_mode = 'F7_Mono8_1288x964_Mode0';
vid = videoinput('pointgrey', 1, Cam_mode);
src = getselectedsource(vid);
mg = 3; %magnification
ulc = 500;
ulr = 750;
vid.FramesPerTrigger = 1;
triggerconfig(vid, 'manual');
vid.TriggerRepeat = Inf;
start(vid);

sta = 1;
fin = 16;

%% capture white
if sta == 1
    input = uint8(imread('../../OneDrive - m.titech.ac.jp/Lab/data/sample_image128/Cameraman.png'));
    input = imresize(input, [n, n]) / 255;
    hadamard_temp = input;

    Line = zeros(wy_pro, wx_pro);

    for i = 1:n

        for j = 1:n
            temp = zeros(mg, mg);
            temp(:) = hadamard_temp(j, i);
            Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg) = temp;
        end

    end

    disp('capture white')
    imshow(Line, 'Parent', ha);

    pause(2)

    trigger(vid);
    img = getdata(vid, 1);

    imwrite(img, ['../../OneDrive - m.titech.ac.jp/Lab/data/capture_', expDate, '/Cameraman.png'], 'BitDepth', 8);

    clear hadamard_temp;
end

%% Stop camera
stop(vid);
delete(vid);
clear vid