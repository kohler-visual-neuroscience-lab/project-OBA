function [p2am,maxWidthAM,maxHeightAM] = Pix2Arcmin(videoMode,parameters)
    % xDiva function by Alexandra Yakovleva to get the number of arcminutes
    % per pixels. 
    % Modified by pjkohler to also compute max height and width in arcminutes. 
    % [p2am,maxWidthAM,maxHeightAM] = xDiva.Pix2Arcmin(videoMode,parameters)
    VMVal = @(x) videoMode{ ismember( videoMode(:,1), {x} ), 2 };
    PVal =  @(x,y) parameters{ismember( { 'S' 'B' '1' '2' }, {x} )}{ismember(parameters{ismember( { 'S' 'B' '1' '2' }, {x})}(:,1),y),2};    
    width_pix = VMVal('widthPix');
    height_pix = VMVal('heightPix');
    width_cm = VMVal('imageWidthCm');
    height_cm = VMVal('imageHeightCm');
    viewDistCm = PVal('S','View Dist (cm)');
    maxWidthAM = 2 * atand( (width_cm/2)/viewDistCm ) * 60;
    maxHeightAM = 2 * atand( (height_cm/2)/viewDistCm ) * 60;
    p2am = max([( maxHeightAM  ) / height_pix,( maxHeightAM  ) / height_pix]); % err on the side of larger conversion values
end

