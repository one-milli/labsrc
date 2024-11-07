%% Main Script: Simulate Pattern Capture and Binary Search using Hadamard Patterns
addpath('common_func_mat');
% Initialize Image Acquisition
initializeImageAcquisition();

% Define Experiment Parameters
params = defineParameters();

% Initialize Figure for Display
[hFig, ha] = initializeFigure(params.rectPro);

% Setup Camera
camera = setupCamera(params);

% Capture White Image
captureWhiteImage(params, camera, ha);

% Capture Hadamard Patterns
captureHadamardPatterns(params, camera, ha);

% Cleanup Camera
cleanupCamera(camera);

%% Function Definitions %%

function captureWhiteImage(params, camera, ha)
    % Capture and save the white image
    if params.sta == 1
        whiteImagePath = fullfile(params.paths.hadamardInput, 'hadamard_1.png');
        inputImage = readAndResizeImage(whiteImagePath, params.n);

        % Create display line for white image
        lineImage = createDisplayLine(inputImage, params);

        disp('Capturing white image...');
        imshow(lineImage, 'Parent', ha);
        pause(2);

        % Trigger camera and capture image
        trigger(camera.vid);
        capturedImg = getdata(camera.vid, 1);

        % Save captured white image
        saveImage(capturedImg, fullfile(params.paths.capturePath, 'capturewhite.png'));
    end

end

function captureHadamardPatterns(params, camera, ha)
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
                ['hadamard_', num2str(patternIndex), '.png']);
            hadamardChunk(:, :, k) = readAndResizeImage(patternPath, params.n);
        end

        % Display each pattern and capture the corresponding image
        for k = 1:currentChunkSize
            patternIndex = chunkStart + k - 1;
            lineImage = createDisplayLine(hadamardChunk(:, :, k), params);

            disp(['Displaying and capturing pattern i = ', num2str(patternIndex)]);
            imshow(lineImage, 'Parent', ha);

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
