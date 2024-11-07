function saveImage(img, filePath)
    % Save the image to the specified file path
    try
        imwrite(img, filePath, 'BitDepth', 8);
    catch ME
        warning('Failed to save image to %s: %s', filePath, ME.message);
    end

end
