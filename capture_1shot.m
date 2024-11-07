%% Main Script: Simulate Snapshot Capture Using a Sample Image
addpath('common_func_mat');
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
