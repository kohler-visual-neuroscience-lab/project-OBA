function G = makeGratingPatch(ppdXY,fCarrier,thetaCarrier,phiCarrier,sigmaXY,envXY,thetaEnv,rectEnv)
% USAGE:
% G = makeGaborPatch(ppdXY,fCarrier,thetaCarrier,phiCarrier,sigmaXY,envXY,thetaEnv,rectEnv)
% 
% INPUTS:
% ppdXY        = pixels/deg
% fCarrier     = carrier frequency (cycle/deg)
% thetaCarrier = carrier orientation (deg)
% phiCarrier   = carrier phase (deg)
% sigmaXY      = Gaussian envelope (deg)
% envXY        = envelope size (deg)  
% thetaEnv     = envelope orientation (deg)
% rectEnv      = logical flag for whether envelope is rectangular or elliptical
%
% OUTPUT:
% G = Gabor pattern (2D double) in range [-1,1]
%
% note:
% ppdXY & sigmaXY & envXY can be either scalar, or 2-element vector for [ horizontal, vertical ]
% thetaCarrier & thetaEnv are defined clockwise
% grating is sine wave in carrier space, i.e. zero in center, rising w/ +X

sinEnv = sind( thetaEnv );
cosEnv = cosd( thetaEnv );

if numel( ppdXY ) == 1
	ppdXY = ppdXY([1 1]);
end
if numel( sigmaXY ) == 1
	sigmaXY = sigmaXY([1 1]);
end
if numel( envXY ) == 1
	envXY = envXY([1 1]);
end

% pixels/deg in envelope coords
envXYppd  = [ hypot(cosEnv*ppdXY(1),sinEnv*ppdXY(2)), hypot(sinEnv*ppdXY(1),cosEnv*ppdXY(2)) ];
envXYpix  = envXY .* envXYppd;		% full width,height of envelope (pix)
envXYhalf = envXY / 2;					% half width,height of envelope (deg)

% Get Width,Height (pixels) of unrotated rectangle that contains the Gaussian envelope
if mod( thetaEnv, 180 ) == 0				% 0 or 180 deg
	nXY = round( envXYpix );
elseif mod( thetaEnv, 180 ) == 90		% 90 or 270 deg
	nXY = round( envXYpix([2 1]) );
else
	% get [ LL UL UR LR ] box corner coordinates after rotation
	cornersXY = zeros(4,2);		% [ LL; UL; UR; LR ] (pix)
	cornersXY(2,:) = envXYpix(2) * [  sinEnv,  cosEnv ];		% XY upper left
	cornersXY(4,:) = envXYpix(1) * [  cosEnv, -sinEnv ];		% XY lower right
	cornersXY(3,:) = cornersXY(2,:) + cornersXY(4,:);			% XY upper right
	% find extrema
	cornerXYmin = min( cornersXY, [], 1 );
	cornerXYmax = max( cornersXY, [], 1 );
	cornerXYmin(:) =  ceil( cornerXYmin );
	cornerXYmax(:) = floor( cornerXYmax );
	nXY = cornerXYmax - cornerXYmin + 1;
end

% Full rectangle, screen coords, in visual angle (deg)
[ Y0, X0 ] = ndgrid( ( (nXY(2):-1:1) - (1+nXY(2))/2 ) / ppdXY(2), ( (1:nXY(1)) - (1+nXY(1))/2 ) / ppdXY(1) );

% Fill in Gabor Pattern 
if mod( thetaEnv, 180 ) == 0				% 0 or 180 deg
	if rectEnv
		G = carrierFcn( X0, Y0 );
	else
		kMask = ( X0 / envXYhalf(1) ).^2 + ( Y0 / envXYhalf(2) ).^2 <= 1;
		G = zeros( nXY([2 1]) );
		G(kMask) = carrierFcn( X0(kMask), Y0(kMask) );
	end
elseif mod( thetaEnv, 180 ) == 90		% 90 or 270 deg
	if rectEnv
		G = envelopeFcn( Y0, X0 ) .* carrierFcn( X0, Y0 );
	else
		kMask = ( Y0 / envXYhalf(1) ).^2 + ( X0 / envXYhalf(2) ).^2 <= 1;
		G = zeros( nXY([2 1]) );
		G(kMask) = carrierFcn( X0(kMask), Y0(kMask) );
	end
else
	% Envelope Coords
	X = X0 * cosEnv - Y0 * sinEnv;
	Y = X0 * sinEnv + Y0 * cosEnv;
	if rectEnv
		kMask = ( abs(X) <= envXYhalf(1) ) & ( abs(Y) <= envXYhalf(2) );
	else
		kMask = ( X / envXYhalf(1) ).^2 + ( Y / envXYhalf(2) ).^2 <= 1;
	end
	G = zeros( nXY([2 1]) );
	G(kMask) = carrierFcn( X0(kMask), Y0(kMask) );
end

return

	function Carrier = carrierFcn( Xscreen, Yscreen )
		% Sinusoidal Grating, along X dimension @ thetaCarrier=0, i.e. vertical bars
		% keeping PowerDiva equation from Gabor2D.cp, I like to add phase rather than subtract.
		% spatial frequency operates on X coordinate of Carrier space
		Carrier = sind( 360*fCarrier * ( Xscreen * cosd( thetaCarrier ) - Yscreen * sind( thetaCarrier ) ) - phiCarrier );
	end

	function Envelope = envelopeFcn( Xmask, Ymask )
		% 2D Gaussian
		Envelope = exp( -( Xmask.^2 / (2*sigmaXY(1)^2) + Ymask.^2 / (2*sigmaXY(2)^2) ) );
	end

end

