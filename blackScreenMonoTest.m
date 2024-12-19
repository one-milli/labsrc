%% Simulate the capture of the patterns and their use for binary search
imaqreset

expDate = '241216';
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
Cam_mode = 'F7_Mono8_1288x964_Mode0';
vid = videoinput('pointgrey', 1, Cam_mode);
src = getselectedsource(vid);
mg = 1; %magnification
ulc = 500;
ulr = 750;
vid.FramesPerTrigger = 1;
triggerconfig(vid, 'manual');
vid.TriggerRepeat = Inf;
start(vid);

%% capture
test_image = ["Black1", "Black2", "Black3", "Black4", "Black5", "White", "Cameraman", "Text", "Clock"];
sta = 1;
fin = 9;
chunk_size = 1;

for chunk_start = sta:chunk_size:fin
    chunk_end = min(chunk_start + chunk_size - 1, fin);

    hadamard = zeros(n, n, chunk_end - chunk_start + 1);

    for k = chunk_start:chunk_end
        filename_r = "../../OneDrive - m.titech.ac.jp/Lab/data/sample_image" + string(n) + "/" + test_image(k) + ".png";
        input = uint8(img2gray(imread(filename_r)));
        % input = imresize(input, [n, n]);
        hadamard(:, :, k - chunk_start + 1) = input;
    end

    for k = chunk_start:chunk_end
        Line = zeros(wy_pro, wx_pro);

        for i = 1:n

            for j = 1:n
                temp = zeros(mg, mg);
                temp(:) = hadamard(j, i, k - chunk_start + 1);
                Line(ulc + (j - 1) * mg + 1:ulc + j * mg, ulr + (i - 1) * mg + 1:ulr + i * mg) = temp;
            end

        end

        disp(['i = ', int2str(k)])

        imshow(Line, 'Parent', ha);

        pause(2)

        trigger(vid);
        img = getdata(vid, 1);
        img = img(trimRowFrom:trimRowTo, trimColFrom:trimColTo);
        img = imresize(img, [m m]);

        filename_w = "../../OneDrive - m.titech.ac.jp/Lab/data/hadamard" + string(n) + "_cap_" + expDate + '/' + test_image(k) + ".png";
        imwrite(img, filename_w, 'BitDepth', 8);
    end

    clear hadamard;
end

%% Stop camera
stop(vid);
delete(vid);
clear vid
