function [hFig, ha] = initializeFigure(rectPro)
    set(0, 'defaultfigureposition', rectPro);
    hFig = figure('Units', 'pixels', ...
        'DockControls', 'off', ...
        'MenuBar', 'none', ...
        'ToolBar', 'none');
    ha = axes('Parent', hFig, 'Units', 'pixels', 'Position', [1, 1, rectPro(3), rectPro(4)]);
end
