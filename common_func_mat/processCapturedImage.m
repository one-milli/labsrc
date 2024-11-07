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
