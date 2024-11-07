%% Main Script: Simulate Pattern Capture and Binary Search using Hadamard Patterns
% Initialize Image Acquisition
initializeImageAcquisition();

% Define Experiment Parameters
params = defineParameters();

% Initialize Figure for Display
figureHandle = initializeFigure(params.rectPro);

% Setup Camera
camera = setupCamera(params);

% Capture White Image
captureWhiteImage(params, camera, figureHandle);

% Capture Hadamard Patterns
captureHadamardPatterns(params, camera, figureHandle);

% Cleanup Camera
cleanupCamera(camera);

%% Function Definitions

function initializeImageAcquisition()
    % Reset image acquisition settings
    imaqreset;
end

function params = defineParameters()
    % Experiment date
    params.expDate = '241107';

    % Trimming parameters
    params.trimRowFrom = 351;
    params.trimRowTo = 850;
    params.trimColFrom = 371;
    params.trimColTo = 870;

    % Hadamard size
    params.n = 128;
    params.nn = params.n ^ 2;

    % Pause settings
    pause('on');

    % Resolutions
    params.resolutions.projector = [1920, 1200];
    params.resolutions.monitor = [1366, 768];
    params.resolutions.camera = [1280, 1024];

    % Logarithmic parameters for projector resolution
    params.Nnumx = round(log2(params.resolutions.projector(1)));
    params.Nnumy = round(log2(params.resolutions.projector(2))) + 1;

    % Screen regions
    params.rectPro = [params.resolutions.monitor(1) + 1, ...
                          params.resolutions.monitor(2) - params.resolutions.projector(2) + 1, ...
                          params.resolutions.projector(1), params.resolutions.projector(2)];
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

    % Capture range
    params.sta = 1;
    params.fin = params.nn;

    % Capture settings
    params.chunkSize = 64;

    % File paths
    basePath = '../../OneDrive - m.titech.ac.jp/Lab/data/';
    params.paths.hadamardInput = fullfile(basePath, ['Hadamard', num2str(params.n), '_input']);
    params.paths.captureWhite = fullfile(basePath, ['capture_', params.expDate]);
    params.paths.hadamardCap = fullfile(basePath, ['hadamard', num2str(params.n), '_cap_', params.expDate]);

    % Create directories if they do not exist
    if ~exist(params.paths.captureWhite, 'dir')
        mkdir(params.paths.captureWhite);
    end

    if ~exist(params.paths.hadamardCap, 'dir')
        mkdir(params.paths.hadamardCap);
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

function captureWhiteImage(params, camera, figureHandle)
    % Capture and save the white image
    if params.sta == 1
        whiteImagePath = fullfile(params.paths.hadamardInput, 'hadamard1.png');
        inputImage = readAndResizeImage(whiteImagePath, params.n);

        % Create display line for white image
        lineImage = createDisplayLine(inputImage, params);

        disp('Capturing white image...');
        imshow(lineImage, 'Parent', figureHandle);
        pause(2);

        % Trigger camera and capture image
        trigger(camera.vid);
        capturedImg = getdata(camera.vid, 1);

        % Save captured white image
        saveImage(capturedImg, fullfile(params.paths.captureWhite, 'capturewhite.png'));
    end

end

function captureHadamardPatterns(params, camera, figureHandle)
    % Capture Hadamard patterns in chunks
    for chunkStart = params.sta:params.chunkSize:params.fin
        chunkEnd = min(chunkStart + params.chunkSize - 1, params.fin);
        currentChunkSize = chunkEnd - chunkStart + 1;

        % Preallocate Hadamard matrix for current chunk
        hadamardChunk = zeros(params.n, params.n, currentChunkSize, 'uint8');

        % Load Hadamard patterns for the current chunk
        for k = 1:currentChunkSize
            patternIndex = chunkStart + k - 1;
            patternPath = fullfile(params.paths.hadamardInput, ...
                ['hadamard', num2str(patternIndex), '.png']);
            hadamardChunk(:, :, k) = readAndResizeImage(patternPath, params.n);
        end

        % Display each pattern and capture the corresponding image
        for k = 1:currentChunkSize
            patternIndex = chunkStart + k - 1;
            lineImage = createDisplayLine(hadamardChunk(:, :, k), params);

            disp(['Displaying and capturing pattern i = ', num2str(patternIndex)]);
            imshow(lineImage, 'Parent', figureHandle);

            % Pause appropriately
            if patternIndex == params.sta
                pause(2);
            else
                pause(1);
            end

            % Trigger camera and capture image
            trigger(camera.vid);
            capturedImg = getdata(camera.vid, 1);

            % Trim and resize captured image
            processedImg = processCapturedImage(capturedImg, params);

            % Save processed image
            savePath = fullfile(params.paths.hadamardCap, ...
                ['hadamard_', num2str(patternIndex), '.png']);
            saveImage(processedImg, savePath);
        end

        % Clear Hadamard chunk from memory
        clear hadamardChunk;
    end

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

function lineImage = createDisplayLine(hadamardPattern, params)
    % Create the display line image based on the Hadamard pattern
    mg = params.camera.magnification;
    ulc = params.camera.upperLeftColumn;
    ulr = params.camera.upperLeftRow;
    wy_pro = params.resolutions.projector(2);
    wx_pro = params.resolutions.projector(1);

    lineImage = zeros(wy_pro, wx_pro, 'uint8');

    for i = 1:params.n

        for j = 1:params.n
            pixelValue = hadamardPattern(j, i);
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
    trimmedImg = capturedImg(params.trimRowFrom:params.trimRowTo, ...
        params.trimColFrom:params.trimColTo);
    processedImg = uint8(imresize(trimmedImg, [256, 256]));
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
