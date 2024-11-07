function lineImage = createDisplayLine(sampleImage, params)
    % Create the display line image based on the sample image
    mg = params.camera.magnification;
    ulc = params.camera.upperLeftColumn;
    ulr = params.camera.upperLeftRow;
    wy_pro = params.resolutions.projector(2);
    wx_pro = params.resolutions.projector(1);

    lineImage = zeros(wy_pro, wx_pro, 'uint8');

    for i = 1:params.n

        for j = 1:params.n
            pixelValue = sampleImage(j, i);
            temp = uint8(pixelValue) * ones(mg, mg, 'uint8');
            rowStart = ulc + (j - 1) * mg + 1;
            rowEnd = ulc + j * mg;
            colStart = ulr + (i - 1) * mg + 1;
            colEnd = ulr + i * mg;
            lineImage(rowStart:rowEnd, colStart:colEnd) = temp;
        end

    end

end
