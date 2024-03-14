rgb = xyz2rgb([0.2148, 0.0073, 1.0391], 'OutputType', 'uint8');

rgbhex = dec2hex(rgb);

colorcode = ['#', rgbhex(1, :), rgbhex(2, :), rgbhex(3, :)];

disp(rgb);
disp(colorcode);
