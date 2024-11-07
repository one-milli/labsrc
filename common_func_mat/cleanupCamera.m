function cleanupCamera(camera)
    % Stop and delete the camera object
    if isvalid(camera.vid)
        stop(camera.vid);
        delete(camera.vid);
    end

    clear camera;
end
