%% Main Script: Simulate Snapshot Capture Using a Sample Image
% Initialize Image Acquisition
initializeImageAcquisition();

% Define Experiment Parameters
params = defineParameters();

% Initialize Figure for Display
figureHandle = initializeFigure(params.rectPro);

% Setup Camera
camera = setupCamera(params);

% Capture Snapshot Image
captureSnapshot(params, camera, figureHandle);

% Cleanup Camera
cleanupCamera(camera);

%% Function Definitions

function initializeImageAcquisition()
    % Reset image acquisition settings
    imaqreset;
end

function params = defineParameters()
    % Define and organize all experiment parameters

    % Experiment date
    params.expDate = '241107';

    % Trimming parameters
    params.trimRowFrom = 351;
    params.trimRowTo = 850;
    params.trimColFrom = 371;
    params.trimColTo = 870;

    % Image size parameters
    params.n = 128;
    params.nn = params.n ^ 2;

    % Pause settings
    pause('on');

    % Resolutions
    params.resolutions.projector = [1920, 1200]; % [width, height]
    params.resolutions.monitor = [1366, 768]; % [width, height]
    params.resolutions.camera = [1280, 1024]; % [width, height]

    % Logarithmic parameters for projector resolution
    params.Nnumx = round(log2(params.resolutions.projector(1)));
    params.Nnumy = round(log2(params.resolutions.projector(2))) + 1; % To cover all pixels

    % Screen regions
    params.rectPro = [params.resolutions.monitor(1) + 1, ...
                          params.resolutions.monitor(2) - params.resolutions.projector(2) + 1, ...
                          params.resolutions.projector(1), params.resolutions.projector(2)]; % Origin (PC lower left)
    params.rectPc = [1, params.resolutions.monitor(2), ...
                         params.resolutions.monitor(1), params.resolutions.monitor(2)];
    params.rectPcConfirm = [floor(params.resolutions.monitor(1) / 4), ...
                                floor(params.resolutions.monitor(2) / 4), ...
                                floor(params.resolutions.monitor(1) / 2), ...
                                floor(params.resolutions.monitor(2) / 2)];

    % Camera settings
    params.camera.mode = 'F7_Mono8_1288x964_Mode0';
    params.camera.deviceID = 1;
    params.camera.magnification = 3;
    params.camera.upperLeftColumn = 500;
    params.camera.upperLeftRow = 750;

    % Image display and capture settings
    params.filename = 'Cameraman';
    params.sampleImagePath = '../../OneDrive - m.titech.ac.jp/Lab/data/sample_image128/';
    params.capturePath = ['../../OneDrive - m.titech.ac.jp/Lab/data/capture_', params.expDate];

    % Ensure capture directory exists
    if ~exist(params.capturePath, 'dir')
        mkdir(params.capturePath);
    end

end

function hFig = initializeFigure(rectPro)
    % Initialize figure with specified position
    set(0, 'defaultfigureposition', rectPro);
    hFig = figure('Units', 'pixels', ...
        'DockControls', 'off', ...
        'MenuBar', 'none', ...
        'ToolBar', 'none');
    axes('Parent', hFig, 'Units', 'pixels', 'Position', [1, 1, rectPro(3), rectPro(4)]);
end

function camera = setupCamera(params)
    % Initialize and configure the camera
    try
        camera.vid = videoinput('pointgrey', params.camera.deviceID, params.camera.mode);
        camera.src = getselectedsource(camera.vid);
        camera.vid.FramesPerTrigger = 1;
        triggerconfig(camera.vid, 'manual');
        camera.vid.TriggerRepeat = Inf;
        start(camera.vid);
    catch ME
        error('Error initializing camera: %s', ME.message);
    end

end

function captureSnapshot(params, camera, figureHandle)
    % Capture and save a snapshot using a sample image

    % Read and resize the sample image
    sampleImagePath = fullfile(params.sampleImagePath, [params.filename, '.png']);
    sampleImage = readAndResizeImage(sampleImagePath, params.n);

    % Create display line image based on the sample image
    lineImage = createDisplayLine(sampleImage, params);

    % Display the image on the projector
    disp('Displaying snapshot image...');
    imshow(uint8(lineImage), 'Parent', figureHandle);
    pause(2); % Wait for the image to stabilize

    % Trigger camera and capture the image
    trigger(camera.vid);
    capturedImg = getdata(camera.vid, 1);

    % Process the captured image (trim and resize)
    processedImg = processCapturedImage(capturedImg, params);

    % Save the processed image
    savePath = fullfile(params.capturePath, [params.filename, '.png']);
    saveImage(processedImg, savePath);

    disp(['Snapshot saved to ', savePath]);
end

function img = readAndResizeImage(filePath, targetSize)
    % Read an image from the specified path and resize it
    try
        img = imread(filePath);
        img = uint8(imresize(img, [targetSize, targetSize]));
    catch ME
        error('Error reading or resizing image %s: %s', filePath, ME.message);
    end

end

function lineImage = createDisplayLine(sampleImage, params)
    % Create the display line image based on the sample image
    mg = params.camera.magnification;
    ulc = params.camera.upperLeftColumn;
    ulr = params.camera.upperLeftRow;
    wy_pro = params.resolutions.projector(2);
    wx_pro = params.resolutions.projector(1);

    lineImage = zeros(wy_pro, wx_pro, 'uint8');

    for i = 1:params.n

        for j = 1:params.n
            pixelValue = sampleImage(j, i);
            temp = uint8(pixelValue) * ones(mg, mg, 'uint8');
            rowStart = ulc + (j - 1) * mg + 1;
            rowEnd = ulc + j * mg;
            colStart = ulr + (i - 1) * mg + 1;
            colEnd = ulr + i * mg;
            lineImage(rowStart:rowEnd, colStart:colEnd) = temp;
        end

    end

end

function processedImg = processCapturedImage(capturedImg, params)
    % Trim and resize the captured image
    try
        trimmedImg = capturedImg(params.trimRowFrom:params.trimRowTo, ...
            params.trimColFrom:params.trimColTo);
        processedImg = uint8(imresize(trimmedImg, [256, 256]));
    catch ME
        error('Error processing captured image: %s', ME.message);
    end

end

function saveImage(img, filePath)
    % Save the image to the specified file path
    try
        imwrite(img, filePath, 'BitDepth', 8);
    catch ME
        warning('Failed to save image to %s: %s', filePath, ME.message);
    end

end

function cleanupCamera(camera)
    % Stop and delete the camera object
    if isvalid(camera.vid)
        stop(camera.vid);
        delete(camera.vid);
    end

    clear camera;
end
