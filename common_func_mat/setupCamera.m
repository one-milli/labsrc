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
