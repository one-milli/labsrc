wl = 488;

rgb = xyz2rgb([0.1655, 0.8620, 0.0422], 'OutputType', 'uint8');

rgbhex = dec2hex(rgb);

colorcode = ['#', rgbhex(1, :), rgbhex(2, :), rgbhex(3, :)];

disp(rgb);
disp(colorcode);
