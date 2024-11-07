function img = readAndResizeImage(filePath, targetSize)
    % Read an image from the specified path and resize it
    try
        img = imread(filePath);
        img = uint8(imresize(img, [targetSize, targetSize]));
    catch ME
        error('Error reading or resizing image %s: %s', filePath, ME.message);
    end

end
