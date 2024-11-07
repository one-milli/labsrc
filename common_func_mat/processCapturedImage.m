function processedImg = processCapturedImage(capturedImg, params)
    % Trim and resize the captured image
    try
        trimmedImg = capturedImg(params.trimRowFrom:params.trimRowTo, ...
            params.trimColFrom:params.trimColTo);
        processedImg = uint8(imresize(trimmedImg, [params.m, params.m]));
    catch ME
        error('Error processing captured image: %s', ME.message);
    end

end
