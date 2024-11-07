function params = defineParameters()
    % Define and organize all experiment parameters

    % Experiment date
    params.expDate = '241107';

    % Trimming parameters
    params.trimRowFrom = 351;
    params.trimRowTo = 850;
    params.trimColFrom = 351;
    params.trimColTo = 850;

    % Image size parameters
    params.m = 255;
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
    params.sampleImagePath = ['../../OneDrive - m.titech.ac.jp/Lab/data/sample_image', num2str(params.n), '/'];
    params.capturePath = ['../../OneDrive - m.titech.ac.jp/Lab/data/capture_', params.expDate];

    % Ensure capture directory exists
    if ~exist(params.capturePath, 'dir')
        mkdir(params.capturePath);
    end

end
