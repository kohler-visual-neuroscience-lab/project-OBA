function pmf_attnTwoTargets_allconds( varargin )
    addpath(genpath('~/code/git/mgl'))
    addpath(genpath('~/code/git/mrTools'))
    dbstop if error
	% to do: temporal phase for square waves, duty cycle, fixation marks

	% _VV_2015_0206 so that 'definitions' are always initialized
	definitions = MakeDefinitions;
    
	parameters	= {};			% always varargin{2}			initialize here for sceop
	timing		= {};			% always varargin{3}
	videoMode	= {};			% always varargin{4}
    oddStepInfo = {};           % always varargin{6}
    previousTimestamp = {};     % always varargin{7}
	iS = 1;		% indices of parts S,B,1,2 in definitions cell array
	iB = 2;
	i1 = 3;
	i2 = 4;
	iSweep = 5;
	iMod   = 6;
% 	iAux   = 7;

	if nargin == 0
		error('No input')
	elseif ismember( varargin{1}, { 'GetDefinitions', 'ValidateParameters', 'MakeMovie', 'ValidateDefinition' } )
		
        
        eval( varargin{1} )		% feval throws 'Undefined function or variable' error.  need to suppress output?
        
	else
		disp( varargin{1} )
		error('Unknown primary input')
    end
    
    
save(fullfile('gaborAttentionDefinitions.mat'), 'definitions', 'parameters', 'timing', 'videoMode', 'oddStepInfo', 'previousTimestamp');
% cd Support_bilateralGaborAttention
% ppath = setPathParadigm;
	return

	function rV = MakeDefinitions
		% - currently implementing 'integer', 'double', 'nominal'
		% - types of the cells in each parameter row
		% - only "standard" type names can be used in the "type" fields
		% - 'nominal' params should have
		%     (a) at least one item
		%		(b) value within the size of the array
		% - all other params (so far) should have empty arrays of items

		rV = { ...

			% - Part 'S' parameters must use standard parameter names
			% - 'Sweep Type' : at least 'Fixed' sweep type must be defined
			% - 'Modulation' : at least 'None' modulation type must be defined
			% - 'Step Type'  : at least 'Lin Stair' type must be defined,
			%                  first 4 step types are reserved, custom step types can only be added after them
			{
				'View Dist (cm)'  70.0            'double'   {}
				'Mean Lum (cd)'   50.0            'double'   {}
				'Fix Point'       'Cross'          'nominal'  { 'None', 'Cross', 'Up Cue', 'Down Cue', 'Left Cue', 'Right Cue', 'Circle' }		% , 'Nonius', 'Block A', 'Block B' not developed yet
				'Sweep Type'      'Fixed'         'nominal'  { 'Fixed', 'Contrast', 'Contrast1', 'Contrast2' }
				'Step Type'       'Lin Stair'     'nominal'  { 'Lin Stair', 'Log Stair', 'Lin Sin', 'Log Sin' }
				'Sweep Start'      1              'double'   {}
				'Sweep End'        90           'double'   {}
				'Modulation'      'Reversal Sqr'  'nominal'  { 'None', 'Reversal Sqr', 'Reversal Sin', 'OnOff Sqr', 'OnOff Sin' }
			}

			% - Part 'B' definition cell must be defined (at least as an empty array)
			{
				'ModInfo'       1.0           'double'   {}
% 				'Target Gamma'  1.8           'double'   {}			% hide this parameter except when loading externally created pictures, movies etc.
			%	'Encoding'      'Bilateral'    'nominal'  { 'Red-Blue', 'Side-by-Side','Bilateral' }
				'Optics'        'One Screen'  'nominal'  { 'One Screen', '2 Mirrors', 'H Pellicle', 'V Pellicle' }
			    'Condition'       1             'integer' {}
                }

			% - Part '1' 'notused' allows to skip creating GUI element in particular column
			%            'Cycle Frames' has to be 1st row
			{
				'Cycle Frames'         6        'integer'  {}
				'Contrast (%)'         7        'double'   {}
				'Spat Freq (cpd)'       2.0        'double'   {}
				'Orientation (deg)'     0        'double'   {}
				'Envelope SD (deg)'     2        'double'   {}
				'Aperture'             'circular'  'nominal'  { 'square', 'circular' }
				'Aperture Size (deg)'   4        'double'   {}
				'Spat Phase (deg)'      0.0        'double'   {}
				'Temp Phase (deg)'      0.0        'double'   {}
				'Duty Cycle'            0.5        'double'   {}
                'X Eccentricity (deg)'  -2.5         'double'    {}
                'Y Eccentricity (deg)'  2.5           'double'   {}
			}

			% - Part '2' ?optional?
			{
				'Cycle Frames'         8        'integer'  {}
				'Contrast (%)'         7        'double'   {}
				'Spat Freq (cpd)'       2.0        'double'   {}
				'Orientation (deg)'    0        'double'   {}
				'Envelope SD (deg)'     2        'double'   {}
				'Aperture'             'circular'  'nominal'  { 'square', 'circular' }
				'Aperture Size (deg)'   4        'double'   {}
				'Spat Phase (deg)'      0.0        'double'   {}
				'Temp Phase (deg)'      0.0        'double'   {}
				'Duty Cycle'            0.5        'double'   {}
                'X Eccentricity (deg)'  2.5       'double'    {}
                'Y Eccentricity (deg)'  2.5          'double'   {}
			}

			% Sweepable parameters
			% The cell array must contain as many rows as there are supported Sweep Types
			% 1st column (Sweep Types) contains Sweep Type as string
			% 2nd column (Stimulus Visiblity) contains one of the following strings,
			% indicating how stimulus visibility changes when corresponding swept parameter value increases:
			%   'constant'   - stimulus visibility stays constant
			%   'increasing' - stimulus visibility increases
			%   'decreasing' - stimulus visibility decreases
			% 3rd column contains a single-row cell array of pairs, where each pair is a single-row cell
			% array of 2 strings: { Part name, Parameter name }
			% If sweep affects only one part, then you only need one { part, param} pair.
			% If it affects both parts, then you need both  pairs, e.g. for "Contrast" below
			% Standard part names: 'S', 'B', '1', '2'
			{
				'Fixed'     'constant'    {}
 				'Contrast'  'increasing'  { { '1', 'Contrast (%)' }, { '2', 'Contrast (%)' } }		% everything in list gets swept
                'Contrast1' 'increasing'  {{'1', 'Contrast (%)'}}
                'Contrast2' 'increasing'  {{'2', 'Contrast (%)'}}
                
                
            }

			% ModInfo information
			% The cell array must contain as many rows as there are supported Modulations
			% 1st column (Modulation) contains one of the supported Modulation typs as string
			% 2nd column contains the name of the ModInfo parameter as string
			% 3rd column (default value) contains default value of the ModInfo
			% parameter for this Modulation
			{
				'None'          'ModInfo'       0.0
				'Reversal Sqr'  'ModInfo'       0.0
				'Reversal Sin'  'ModInfo'       0.0
				'OnOff Sqr'     'OffLum (Cd)'  50.0
				'OnOff Sin'     'OffLum (Cd)'  50.0
			}

			% Auxiliary information.
			% Required by xDiva, but not by Matlab Function
			{
				'Version'                   1							% of the matlab function?
				'Adjustable'                true
				'Needs Unique Stimuli'      true					% e.g. if you want to re-randomize something every trial set to true
				'Supports Interleaving'     true
				'Part Name'                 { 'Left' 'Right'}
				'Frame Rate Divisor'        { 1 1 }
				'Max Cycle Frames'          { 60 60 }
				'Allow Static Part'         { false false }
% 				'Supported Psy Procedures'  {}						% currently not used
% 				'Supported Psy Nulls'       {}						% currently not used
			}

		};

	end

	function GetDefinitions
		assignin( 'base', 'output', definitions );
	end

	function ValidateParameters
		% xDiva invokes Matlab Engine command:
		% pmf_<subParadigmName>( 'ValidateParameters', parameters, timing, videoMode );
		% "parameters" here is an input argument. Its cellarray has the same structure
		% as "definitions" but each parameter row has only first two elements

		% The "timing" and "videoMode" cellarrays have the same row
		% structure with each row having a "name" and "value" elements.

		% timing names from TimingRec declaration in PowerDiva.h
		% videoMode names from VideoModeRx declaration in VxVideoMode.h

		% nFrameCycle >= 2 if square wave
		%              = 1x or more multiple of 4 if sine wave
		[ parameters, timing, videoMode ] = deal( varargin{2:4} );				% parameters only contains 1st 4 cells???
        save('oba_params.mat','parameters', 'timing','videoMode');
        
        nFrameCycle1 = GrabCellValue( parameters{i1}, 'Cycle Frames' );		% double
		nFrameCycle2 = GrabCellValue( parameters{i2}, 'Cycle Frames' );
		% validate OffLum too if I can get at it!
		parValidMsgs = {
			'frames/cycle1 less than 2'								% 1
			'frames/cycle2 less than 2'								% 2
			'frames/cycle1 not multiple of 4'						% 3
			'frames/cycle2 not multiple of 4'						% 4
			'mean luminance out of range'								% 5
			'contrast1 exceeds luminance range'						% 6
			'contrast2 exceeds luminance range'						% 7
			'gabor1 exceeds screen dimensions'						% 8
			'gabor2 exceeds screen dimensions'						% 9
			'gabor1 too small'											% 10
			'gabor2 too small'											% 11
			'odd pixel width in side-by-side configuration'		% 12
			'unachievable square wave phase1'						% 13
			'unachievable square wave phase2'						% 14
			'unachievable duty cycle1'									% 15
			'unachievable duty cycle2'									% 16
		};
		parValidFlags = true(size(parValidMsgs));
		switch GrabCellValue( parameters{iS}, 'Modulation' )
		case 'None'
		case { 'Reversal Sqr', 'OnOff Sqr' }
			parValidFlags(1) = nFrameCycle1 >= 2;
			parValidFlags(2) = nFrameCycle2 >= 2;
			parReq = GrabCellValue( parameters{i1}, 'Temp Phase (deg)' );				% requested value
			parAct = round( parReq / 360 * nFrameCycle1 ) * 360 / nFrameCycle1;		% actual value
			parValidFlags(13) = parAct == parReq;
			if ~parValidFlags(13)
				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Temp Phase (deg)' ), 2 } = parAct;
				parValidMsgs{13} = [ parValidMsgs{13}, sprintf(' set to %0.1f deg',parAct) ];
			end
			parReq = GrabCellValue( parameters{i2}, 'Temp Phase (deg)' );
			parAct = round( parReq / 360 * nFrameCycle2 ) * 360 / nFrameCycle2;
			parValidFlags(14) = parAct == parReq;
			if ~parValidFlags(14)
				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Temp Phase (deg)' ), 2 } = parAct;
				parValidMsgs{14} = [ parValidMsgs{14}, sprintf(' set to %0.1f deg',parAct) ];
			end
			parReq = GrabCellValue( parameters{i1}, 'Duty Cycle' );
			parAct = min( max( round( parReq * nFrameCycle1 ), 1 ), nFrameCycle1-1 ) / nFrameCycle1;
			parValidFlags(15) = parAct == parReq;
			if ~parValidFlags(15)
				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Duty Cycle' ), 2 } = parAct;
				parValidMsgs{15} = [ parValidMsgs{15}, sprintf(' set to %0.3f',parAct) ];
			end
			parReq = GrabCellValue( parameters{i2}, 'Duty Cycle' );
			parAct = min( max( round( parReq * nFrameCycle2 ), 1 ), nFrameCycle2-1 ) / nFrameCycle2;
			parValidFlags(16) = parAct == parReq;
			if ~parValidFlags(16)
				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Duty Cycle' ), 2 } = parAct;
				parValidMsgs{16} = [ parValidMsgs{16}, sprintf(' set to %0.3f',parAct) ];
			end
		case 'Reversal Sin'
			parValidFlags(3) = mod(nFrameCycle1,4) == 0;
			if ~parValidFlags(3)
				nFrameCycle1(:) = round( nFrameCycle1 / 4 ) * 4;
				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Cycle Frames' ), 2 } = nFrameCycle1;
				parValidMsgs{3} = [ parValidMsgs{3}, sprintf(' set to %d',nFrameCycle1) ];
			end
			parValidFlags(4) = mod(nFrameCycle2,4) == 0;
			if ~parValidFlags(4)
				nFrameCycle2(:) = round( nFrameCycle2 / 4 ) * 4;
				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Cycle Frames' ), 2 } = nFrameCycle2;
				parValidMsgs{4} = [ parValidMsgs{4}, sprintf(' set to %d',nFrameCycle2) ];
			end
		case 'OnOff Sin'
% 			parValidFlags(3) = mod(nFrameCycle1,4) == 0;
% 			if ~parValidFlags(3)
% 				nFrameCycle1(:) = round( nFrameCycle1 / 4 ) * 4;
% 				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Cycle Frames' ), 2 } = nFrameCycle1;
% 				parValidMsgs{3} = [ parValidMsgs{3}, sprintf(' set to %d',nFrameCycle1) ];
% 			end
% 			parValidFlags(4) = mod(nFrameCycle2,4) == 0;
% 			if ~parValidFlags(4)
% 				nFrameCycle2(:) = round( nFrameCycle2 / 4 ) * 4;
% 				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Cycle Frames' ), 2 } = nFrameCycle2;
% 				parValidMsgs{4} = [ parValidMsgs{4}, sprintf(' set to %d',nFrameCycle2) ];
% 			end
			parReq = GrabCellValue( parameters{i1}, 'Temp Phase (deg)' );				% requested value
			parAct = round( parReq / 360 * nFrameCycle1 ) * 360 / nFrameCycle1;		% actual value
			parValidFlags(13) = parAct == parReq;
			if ~parValidFlags(13)
				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Temp Phase (deg)' ), 2 } = parAct;
				parValidMsgs{13} = [ parValidMsgs{13}, sprintf(' set to %0.1f deg',parAct) ];
			end
			parReq = GrabCellValue( parameters{i2}, 'Temp Phase (deg)' );
			parAct = round( parReq / 360 * nFrameCycle2 ) * 360 / nFrameCycle2;
			parValidFlags(14) = parAct == parReq;
			if ~parValidFlags(14)
				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Temp Phase (deg)' ), 2 } = parAct;
				parValidMsgs{14} = [ parValidMsgs{14}, sprintf(' set to %0.1f deg',parAct) ];
			end
			parReq = GrabCellValue( parameters{i1}, 'Duty Cycle' );
			parAct = min( max( round( parReq * nFrameCycle1 / 4 ) * 4, 4 ), nFrameCycle1 ) / nFrameCycle1;
			parValidFlags(15) = parAct == parReq;
			if ~parValidFlags(15)
				parameters{i1}{ strcmp( parameters{i1}(:,1), 'Duty Cycle' ), 2 } = parAct;
				parValidMsgs{15} = [ parValidMsgs{15}, sprintf(' set to %0.3f',parAct) ];
			end
			parReq = GrabCellValue( parameters{i2}, 'Duty Cycle' );
			parAct = min( max( round( parReq * nFrameCycle2 / 4 ) * 4, 4 ), nFrameCycle2 ) / nFrameCycle2;
			parValidFlags(16) = parAct == parReq;
			if ~parValidFlags(16)
				parameters{i2}{ strcmp( parameters{i2}(:,1), 'Duty Cycle' ), 2 } = parAct;
				parValidMsgs{16} = [ parValidMsgs{16}, sprintf(' set to %0.3f',parAct) ];
			end
		end
		% check that contrasts don't drive luminance out of range
% 		lumMean = GrabCellValue( videoMode, 'meanLuminanceCd' );
		lumMean = GrabCellValue( parameters{iS}, 'Mean Lum (cd)' );
		lumMin  = GrabCellValue( videoMode, 'minLuminanceCd' );
		lumMax  = GrabCellValue( videoMode, 'maxLuminanceCd' );
		parValidFlags(5) = lumMean >= lumMin && lumMean <= lumMax;
		if ~parValidFlags(5)
			% Vladimir said set it to display's mean, not offending limit
			lumMean =  GrabCellValue( videoMode, 'meanLuminanceCd' );
			parameters{iS}{ strcmp( parameters{iS}(:,1), 'Mean Lum (cd)' ), 2 } = lumMean;
			parValidMsgs{5} = [ parValidMsgs{5}, ' set to videoMode mean' ];
		end
		contrastMax = min( 1 - lumMin/lumMean, lumMax/lumMean - 1 );
		contrast = GrabCellValue( parameters{i1}, 'Contrast (%)' ) / 100;
		parValidFlags(6) = contrast <= contrastMax && contrast >= 0;
		if ~parValidFlags(6)
			parameters{i1}{ strcmp( parameters{i1}(:,1), 'Contrast (%)' ), 2 } = contrastMax * 100;			
			parValidMsgs{6} = [ parValidMsgs{6}, sprintf(' set to %0.1f%%',contrastMax*100) ];
		end
		contrast = GrabCellValue( parameters{i2}, 'Contrast (%)' ) / 100;
		parValidFlags(7) = contrast <= contrastMax && contrast >= 0;
		if ~parValidFlags(7)
			parameters{i2}{ strcmp( parameters{i2}(:,1), 'Contrast (%)' ), 2 } = contrastMax * 100;			
			parValidMsgs{7} = [ parValidMsgs{7}, sprintf(' set to %0.1f%%',contrastMax*100) ];
		end
	
		% check that patches fit on screen and are not too small to have any meaningful content
		wPx = GrabCellValue( videoMode, 'widthPix' );
		hPx = GrabCellValue( videoMode, 'heightPix' );
		wCm = GrabCellValue( videoMode, 'imageWidthCm' );
		hCm = GrabCellValue( videoMode, 'imageHeightCm' );
		dCm = GrabCellValue( parameters{iS}, 'View Dist (cm)' );
		sideBySide  =0;%  strcmp( GrabCellValue( parameters{iB}, 'Encoding' ), 'Side-by-Side' );
        bilateral  = 1;% strcmp( GrabCellValue( parameters{iB}, 'Encoding' ), 'Bilateral' );
        
		multiScreen = ~strcmp( GrabCellValue( parameters{iB}, 'Optics' ), 'One Screen' );
		if sideBySide
			parValidFlags(12) = mod( wPx, 2 ) == 0;
			if multiScreen
				ppdXY  = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm );
				%wPx = wPx / 2;
			else
				%wPx = wPx / 2;
				ppdXY  = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm );
			end
		else
			ppdXY  = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm );
        end
        if bilateral
            parValidFlags(12) = mod( wPx, 2 ) == 0;
        end
		gaborMin = 5;		% minimum edge (pixels) of Gabor patch
		pixXY = round( GrabCellValue( parameters{i1}, 'Aperture Size (deg)' ) * ppdXY );
		parValidFlags(8) = all( pixXY <= [ wPx, hPx ] );
		parValidFlags(10) = all( pixXY >= gaborMin );
		pixXY = round( GrabCellValue( parameters{i2}, 'Aperture Size (deg)' ) * ppdXY );
		parValidFlags(9) = all( pixXY <= [ wPx, hPx ] );
		parValidFlags(11) = all( pixXY >= gaborMin );
		% finish
		parValidFlag = all( parValidFlags );
		if parValidFlag
			parValidMsg = { 'OK' };
		else
			parValidMsg = parValidMsgs( ~parValidFlags );
		end
		assignin( 'base', 'output', { parValidFlag, parameters, parValidMsg } )
	end
 
    function ValidateDefinition
		% Test to make sure that 'definitions' array is correct, i.e.
		% correct number of parts, structure of param items, etc. xDiva
		% never makes a call with 'ValidateDefinition' input
		% This is for the Matlab user during paradigm development.

		% check for specific names & order in definitions{iS}(:,1)!
		defValidFlag = numel( definitions ) == 7;
		% check that the current value of all nominal parameters is a member of the list
		for iPart = 1:numel( definitions )
			kNominal = strcmp( definitions{iPart}(:,3), 'nominal' );
			if any(kNominal)
				for iRow = find(kNominal)'
					defValidFlag = defValidFlag && ismember( definitions{iPart}{iRow,2}, definitions{iPart}{iRow,4} );
				end
			end
		end
		% check that all defined sweep types are described
		list1        = GrabCellValue( definitions{iS}, 'Sweep Type', 4 );
		list2        = definitions{iSweep}(:,1)';
		defValidFlag = defValidFlag && ( numel( list1 ) == numel( list2 ) ) && all( ismember( list1, list2 ) );
		% check that all defined modulations are described
		list1        = GrabCellValue( definitions{iS}, 'Modulation', 4 );
		list2        = definitions{iMod}(:,1)';
		defValidFlag = defValidFlag && ( numel( list1 ) == numel( list2 ) ) && all( ismember( list1, list2 ) );
		% check that part1 & part2 have same parameter list
		list1        = definitions{i1}(:,1);
		list2        = definitions{i1}(:,2);
		defValidFlag = defValidFlag && ( numel( list1 ) == numel( list2 ) ) && all( strcmp( list1, list2 ) );
		assignin( 'base', 'validDefinition', defValidFlag );
    end
 
    function MakeMovie
		% xDiva creates variables "parameters", "timing", "videoMode" in
		% Matlab Engine workspace and invokes Matlab Engine command:

		% pmf_<ParadigmName>( 'MakeMovie', parameters, timing, videoMode );
		% where pmf_<ParadigmName> is the name of the m-file selected by
		% the MatlabFunction paradigm dialog "Choose" control

		% note: Don't bother with fixation point in actual movie, xDiva will add it later.

        
		try
           
% 			[ parameters, timing, videoMode, trialNumber ] = deal( varargin{2:5} );
% 			[ parameters, timing, videoMode ] = deal( varargin{2:4} );
            [ parameters, timing, videoMode, trialNumber, oddStepInfo, previousTimestamp] = deal( varargin{2:7} );
            
             % before making movie, load previous timestamps and key presses
            mglListener('init');  
            key = mglListener('getAllKeyEvents');
%             keycode=mglGetKeys;  %rta
%             key.keyCode=find(keycode);  %rta
            if isempty(key)
                key.when = nan; key.keyCode = nan;
            end
        if exist('keyRecord.mat','file') == 2
            load('keyRecord.mat');         
        else
            keyFile.key.when = []; keyFile.key.keyCode=[]; keyFile.key.rt = [];
        end
            keyFile.key.when = [keyFile.key.when, key.when];
            keyFile.key.keyCode = [keyFile.key.keyCode, key.keyCode];
            thisRT = key.when - previousTimestamp;
            keyFile.key.rt = [keyFile.key.rt, thisRT];
        
        save(fullfile('keyRecord.mat'), 'keyFile');
        
 
            oddParams.hasOddStepTask = GrabCellValue(oddStepInfo, 'hasOddStepTask');
            if oddParams.hasOddStepTask
            oddParams.oddParams = GrabCellValue(oddStepInfo, 'oddParams');
            oddParams.oddSteps = GrabCellValue(oddStepInfo, 'oddSteps');
            end
            if oddParams.hasOddStepTask
                if size(oddParams.oddParams, 1) == 2
                    oddParams.oddCurrent = oddParams.oddParams;
                    oddParams.oddLocation = 'Both';
                elseif size(oddParams.oddParams, 1) == 1
                    if oddParams.oddParams(1,1) == 1 % odd Left
                        oddParams.oddCurrent = oddParams.oddParams;
%                         oddParams.oddCurrent(2,:) = [nan nan nan];
                        oddParams.oddLocation = 1;
                    else % odd Right
                        oddParams.oddCurrent = oddParams.oddParams;
%                         oddParams.oddCurrent(1,:) = [nan nan nan];
                        oddParams.oddLocation = 2;
                    end
                end
            end

% 			nBinCore      = GrabCellValue( timing, 'nmbCoreBins' );
			timingParams.nFrameBin     = GrabCellValue( timing, 'nmbFramesPerBin' );
			timingParams.nFrameStep    = GrabCellValue( timing, 'nmbFramesPerStep' );
			timingParams.nBinPrelude   = GrabCellValue( timing, 'nmbPreludeBins' );
			timingParams.nStepCore     = GrabCellValue( timing, 'nmbCoreSteps' );
			timingParams.nFramePrelude = timingParams.nFrameBin * timingParams.nBinPrelude;
			timingParams.nFrameCore    = timingParams.nFrameStep * timingParams.nStepCore;
			timingParams.nFrameTrial   = timingParams.nFrameCore + 2 * timingParams.nFramePrelude;
            
         currCond=GrabCellValue( parameters{iB}, 'Condition' );

            targetPresent = round(rand);
            nontargetPresent = round(rand);
            if currCond==3
                targetornot=rand;
                if targetornot<.5
                    targetPresent=0;
                    nontargetPresent=0;
                else
                    targets=randi(2,1);
                   switch targets
                       case 1
                            targetPresent=1;
                    nontargetPresent=0; 
                       case 2 
                             targetPresent=0;
                    nontargetPresent=1;
                 
                   end
                end
            end
            targetNFrames = 16;
            
            if targetPresent
                targetStart = randi([timingParams.nFrameBin*2+1 timingParams.nFrameTrial-timingParams.nFrameBin*2]);
                targetFrames = targetStart:targetStart+targetNFrames - 1;
            end
            if nontargetPresent
                nontargetStart = randi([timingParams.nFrameBin*2+1 timingParams.nFrameTrial-timingParams.nFrameBin*2]);
                nontargetFrames = nontargetStart:nontargetStart+targetNFrames - 1;
            end
            overlap=[];
            if targetPresent && nontargetPresent
                overlap = intersect(targetFrames,nontargetFrames);
                targetUnique = setdiff(targetFrames, overlap);
                nontargetUnique = setdiff(nontargetFrames, overlap);
            elseif targetPresent && ~nontargetPresent
                targetUnique = targetFrames;
                nontargetUnique = [];
            elseif ~targetPresent && nontargetPresent
                targetUnique = [];
                nontargetUnique = nontargetFrames;
            else
                overlap = [];
                targetUnique = [];
                nontargetUnique = [];
                    end
            %%5
        priors(3,:) = [3.5, 7, 14, 21, 28]; % distributed
        priors(1,:) = [0.88, 1.75, 3.5, 4.5, 6]; %focal -left
%         priors(2,:) = [0.88, 1.75, 3.5, 4.5, 6]; %focal -right
        priors(1,:) = [1, 2, 4, 5, 6.5]*2; %focal -left
        priors(2,:) = [1, 2, 4, 5, 6.5]*2; %focal -right
        %RTA
        contrasts = [3.5, 7, 14, 28, 56]; %rta
        currContrast=GrabCellValue( parameters{i1}, 'Contrast (%)' );
         currCond=GrabCellValue( parameters{iB}, 'Condition' );
       contrastindex=find(contrasts==currContrast);
        cond = currCond;  %rta
  
            targetindex=1;
            nontargetindex=2;
           save tempvars;
           ykey=124;
           nkey=126;
        %/
        %Previous trial correct or incorrect and update staircase
        if exist('staircase.mat','file') == 2
            load('staircase.mat');
            isFirstTrial = 0;
             switch trial.cond(end)
                 case 1
                    if (trial.targetPresent(end) && keyFile.key.keyCode(end)==ykey) || (~trial.targetPresent(end) && keyFile.key.keyCode(end)==nkey)
                        correct = 1;
                    else
                        correct = 0;
                    end
                 case 2             
                     if (trial.nontargetPresent(end) && keyFile.key.keyCode(end)==ykey) || (~trial.nontargetPresent(end) && keyFile.key.keyCode(end)==nkey)
                        correct = 1;
                    else
                        correct = 0;
                     end                   
                
                 case 3
                  if ((trial.targetPresent(end) || trial.nontargetPresent(end))  && keyFile.key.keyCode(end)==ykey) || ((~trial.targetPresent(end) || ~trial.targetPresent(end)) && keyFile.key.keyCode(end)==nkey)
                    correct = 1;
                  else
                    correct = 0;
                  end  
              end
%              if trial.cond(end) == 1
%                 correctbycond(cond) = correct;
%                 correctDistributed = [];
%             else
%                 correctDistributed = correct;
%                 correctFocal = [];
%                         end
            
            save tempvars;
%             stair{currCond}{contrastindex} = doStaircase('update', stair{currCond}{contrastindex}, correct);
%             [testValue, stair{currCond}{contrastindex}] = doStaircase('testValue',stair{currCond}{contrastindex});
%             
%             
            prevcond=trial.cond(end);
            prevcontrast=trial.contrast(end);
            trial.correct = [trial.correct, correct];
            trial.targetPresent = [trial.targetPresent, targetPresent];
            trial.nontargetPresent = [trial.nontargetPresent, nontargetPresent];
            trial.cond = [trial.cond, cond];
            trial.contrast= [trial.contrast, contrastindex];
            trial.correctbycond{prevcond,prevcontrast} = [trial.correctbycond{prevcond,prevcontrast} correct];
%                if ~isempty(trial.correctbycond{cond,contrastindex})
                    stair{prevcond}{prevcontrast} = doStaircase('update', stair{prevcond}{prevcontrast}, trial.correctbycond{prevcond,prevcontrast}(end));
%                 else
%                     if ~isempty(trial.correctFocal)
%                         stair{cond}{contrastindex} = doStaircase('update', stair{{cond}{contrastindex}, trial.correctFocal(end));
%                     end
%                end
             
             [testValue, stair{cond}{contrastindex}] = doStaircase('testValue',stair{cond}{contrastindex});
                trial.testValue = [trial.testValue, testValue];
       
        else
            isFirstTrial = 1;
            for iCond = 1:3 % cue condition
                for iContrast = 1:5
                    stair{iCond}{iContrast} = doStaircase('init','upDown',...
                    'initialThreshold', priors(iCond,iContrast),...
                    'initialStepsize', priors(iCond,iContrast)/2, ...
                    'minStepsize', 0.1,...
                    'maxStepsize', 5,...
                    'minThreshold', 0.1,...
                    'maxThreshold', 100,...
                    'verbose', 0,...
                    'stepRule', 'levitt');
                end
            end
            trial.targetPresent = targetPresent;
            trial.nontargetPresent = nontargetPresent;
            trial.correct = [];
            trial.correctbycond=cell(3,5);
            trial.cond = cond;
            trial.contrast=contrastindex;
            testValue = priors(cond,contrastindex);
            trial.testValue = testValue;
        end
           
            save(fullfile('staircase.mat'), 'stair','trial');

			stimParams.modType = GrabCellValue( parameters{iS}, 'Modulation' );

% 			videoMode items: nominalFrameRateHz, widthPix, heightPix, imageWidthCm, imageHeightCm, minLuminanceCd, maxLuminanceCd, meanLuminanceCd, meanLuminanceBitmapValue, bitsPerPixel, componentCount, bitsPerComponent, gammaTableCapacity, isInterlaced
			videoParams.wPx    =   GrabCellValue( videoMode, 'widthPix' );
			videoParams.hPx    =   GrabCellValue( videoMode, 'heightPix' );
            if mod(videoParams.wPx, 4) % not a multiple of 4
                videoParams.wPx = videoParams.wPx - mod(videoParams.wPx,4);
            end
            if mod(videoParams.hPx, 4)
                videoParams.hPx = videoParams.hPx - mod(videoParams.hPx,4);
            end
			videoParams.wCm    =   GrabCellValue( videoMode, 'imageWidthCm' );
			videoParams.hCm    =   GrabCellValue( videoMode, 'imageHeightCm' );
			videoParams.lumMin =   GrabCellValue( videoMode, 'minLuminanceCd' );
			videoParams.lumMax =   GrabCellValue( videoMode, 'maxLuminanceCd' );
% 			lumBg  =   GrabCellValue( videoMode, 'meanLuminanceCd' );
			videoParams.lumBg  =   GrabCellValue( parameters{iS}, 'Mean Lum (cd)' );
			videoParams.lumRes = 2^GrabCellValue( videoMode, 'bitsPerComponent' );
			videoParams.dCm    =   GrabCellValue( parameters{iS}, 'View Dist (cm)' );
			sideBySide  =0;%  strcmp( GrabCellValue( parameters{iB}, 'Encoding' ), 'Side-by-Side' );
			videoParams.multiScreen = ~strcmp( GrabCellValue( parameters{iB}, 'Optics' ), 'One Screen' );
           bilateral  =1;%  strcmp( GrabCellValue( parameters{iB}, 'Encoding' ), 'Bilateral' );
          
% 			if sideBySide
% 				if multiScreen
% 					ppdXY  = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm );
% 					%wPx = wPx / 2;
% 				else
% 					%wPx = wPx / 2;
% 					ppdXY  = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm );
% 				end
% 			else
				videoParams.ppdXY  = getPixelsPerDeg( videoParams.wPx, videoParams.hPx, videoParams.wCm, videoParams.hCm, videoParams.dCm );
% 			end

			[stimParams.nFrameCycle,stimParams.contrast,stimParams.fSpatial,stimParams.sigma,stimParams.aperture,stimParams.theta,stimParams.phiSpatial,stimParams.phiTemporal,stimParams.dutyCycle] = deal( zeros(1,2) );
			rectAperture = false(1,2);
			I = [ i1, i2 ];
			for iEye = 1:2
				stimParams.nFrameCycle(iEye)  =         GrabCellValue( parameters{I(iEye)}, 'Cycle Frames' );			% double
				stimParams.contrast(iEye)     =         GrabCellValue( parameters{I(iEye)}, 'Contrast (%)' ) / 100;
				stimParams.fSpatial(iEye)     =         GrabCellValue( parameters{I(iEye)}, 'Spat Freq (cpd)' );
				stimParams.theta(iEye)        =         GrabCellValue( parameters{I(iEye)}, 'Orientation (deg)' );
				stimParams.sigma(iEye)        =         GrabCellValue( parameters{I(iEye)}, 'Envelope SD (deg)' );
				stimParams.rectAperture(iEye) = strcmp( GrabCellValue( parameters{I(iEye)}, 'Aperture' ), 'square' );
				stimParams.aperture(iEye)     =         GrabCellValue( parameters{I(iEye)}, 'Aperture Size (deg)' );
				stimParams.phiSpatial(iEye)   =         GrabCellValue( parameters{I(iEye)}, 'Spat Phase (deg)' );
				stimParams.phiTemporal(iEye)  =         GrabCellValue( parameters{I(iEye)}, 'Temp Phase (deg)' );
				stimParams.dutyCycle(iEye)    =         GrabCellValue( parameters{I(iEye)}, 'Duty Cycle' );
                stimParams.yEcc(iEye)      =        GrabCellValue( parameters{I(iEye)}, 'Y Eccentricity (deg)' );
                stimParams.xEcc(iEye)      =        GrabCellValue( parameters{I(iEye)}, 'X Eccentricity (deg)' );
			end
			switch stimParams.modType
			case { 'Reversal Sqr', 'OnOff Sqr' }
				stimParams.dutyFrames = round( stimParams.dutyCycle .* stimParams.nFrameCycle );
			case 'OnOff Sin'
				stimParams.dutyFrames = round( stimParams.dutyCycle .* stimParams.nFrameCycle / 4 ) * 4;
			end
			stimParams.nFramePhi = stimParams.phiTemporal .* stimParams.nFrameCycle/360;
	
			Gabor = cell(1,2);
            GaborOdd = cell(1,2);
			for iEye = find( stimParams.contrast ~= 0 )
                %RTA randomize orientation angle
                orientation1=round(rand(1)*360);%stimParams.theta(iEye)
               
				GaborTemp{iEye} = makeGratingPatch( videoParams.ppdXY, stimParams.fSpatial(iEye), orientation1, ...
                    stimParams.phiSpatial(iEye), stimParams.sigma(iEye), stimParams.aperture(iEye), 0, stimParams.rectAperture(iEye) );
				GaborTemp2{iEye} = makeGratingPatch( videoParams.ppdXY, stimParams.fSpatial(iEye), orientation1+90, ...
                    stimParams.phiSpatial(iEye), stimParams.sigma(iEye), stimParams.aperture(iEye), 0, stimParams.rectAperture(iEye) );
				CheckerTemp{iEye} = sign(GaborTemp{iEye}/2+GaborTemp2{iEye}/2);
                % convert nominal values to delta luminance (Cd)
				Gabor{iEye} = stimParams.contrast(iEye)*videoParams.lumBg * CheckerTemp{iEye};
                

                
            end
            %TARGETS ML RTA
            if targetPresent
            % left target
            GaborOdd{targetindex} = (stimParams.contrast(1) + testValue/100) * videoParams.lumBg * CheckerTemp{targetindex};
            else 
                GaborOdd{targetindex} = Gabor{targetindex};
            end
            % right nontarget
            if nontargetPresent
            GaborOdd{nontargetindex} = (stimParams.contrast(2) + testValue/100) * videoParams.lumBg * CheckerTemp{nontargetindex}; %RTA
%             GaborOdd{nontargetindex} = (stimParams.contrast(2) +
%             priors(1,2)/100) * videoParams.lumBg * CheckerTemp{nontargetindex}; 
            
            else
                GaborOdd{nontargetindex} = Gabor{nontargetindex};
            end
            

			timingParams.preludeType = GrabCellValue( timing, 'preludeType' );			% 0=dynamic, 1=blank, 2=static
			blankFrames = double( timingParams.preludeType == 1 );							% should blank frames use mean lum & not off lum?

            stimParams.valOff = lum2valFcn( GrabCellValue( parameters{iB}, 'ModInfo' ) );
            save(fullfile('xDivaOutputAttention.mat'), ... %ppath.home
            'stimParams', 'videoParams', 'timingParams', 'oddStepInfo','previousTimestamp');
			switch stimParams.modType
			case 'None'									% ++
				nImg = 1 + blankFrames;
			case { 'Reversal Sqr', 'OnOff Sqr' }		% [ ++, +0, 0+, 00 ] or [ ++, +-, -+, -- ]
				nImg = 4 + blankFrames;
			case { 'Reversal Sin', 'OnOff Sin' }
				nImg = lcm( stimParams.nFrameCycle(1), stimParams.nFrameCycle(2) ) + blankFrames;
            end
            if oddParams.hasOddStepTask
                nImg = nImg + 4;
            end
			img2D = zeros( [ videoParams.hPx videoParams.wPx ], 'uint8' );
            img2D_odd = zeros( [ videoParams.hPx videoParams.wPx ], 'uint8' );
			img4D = zeros( [ videoParams.hPx videoParams.wPx 3 nImg ], 'uint8' );
			valBg = lum2valFcn( videoParams.lumBg );
	
			kEye = false( [ videoParams.hPx videoParams.wPx ] );		% mask
			chanEye = [ 1 3 ];					% rgb channel index that each eye goes to, put in a parameter to control this?
			if ismember( stimParams.modType, { 'OnOff Sqr', 'OnOff Sin' } )
% 				valOff = lum2valFcn( GrabCellValue( parameters{iB}, 'ModInfo' ) );
% 				img4D(:,:,chanEye,1:nImg-blankFrames) = valOff;
				for iEye = 1:2
					if stimParams.contrast(iEye) ~= 0
						img2D(:) = valBg;
                        img2D_odd(:) = valBg;
						setRectMask
% 						img2D(kEye) = valOff;
%                         img2D_odd(kEye) = valOff;
						img4D(:,:,chanEye(iEye),1:nImg-blankFrames) = repmat( img2D, [ 1 1 1 nImg-blankFrames ] );
					end
				end
			end
			if blankFrames == 1
				img4D(:,:,chanEye(iEye),nImg) = valBg;
			end
% 				img4D(:,:,setdiff(1:3,chanEye),:) = valBg;			% not necessary, make background gray instead of magenta while debugging
			
			for iEye = 1:2
				if stimParams.contrast(iEye) == 0
					img4D(:,:,chanEye(iEye),1:nImg-blankFrames) = valBg;
				else
					img2D(:) = valBg;
                    img2D_odd(:) = valBg;
				%	setRectMask						% sets kEye mask to center patch on screen
                %RTA 9-2021
                yEccPix=round(stimParams.yEcc*videoParams.ppdXY(1));
                xEccPix=round(stimParams.xEcc*videoParams.ppdXY(1));
            [ hRect, wRect ] = size( Gabor{iEye} );
			yUL = floor( ( videoParams.hPx - hRect ) / 2 );
			xUL = floor( ( videoParams.wPx - wRect ) / 2 );
            
			kEye(:) = false;
			kEye(yUL+1+yEccPix(iEye):yUL+hRect+yEccPix(iEye),xUL+1+xEccPix(iEye):xUL+wRect+xEccPix(iEye)) = true;
            keyes{iEye}=kEye;
 					% [ ++, +-, -+, -- ]
						img2D(kEye) = lum2valFcn( videoParams.lumBg + Gabor{iEye} );
%                         if iEye ==2 ; keyboard; end
						if iEye == 1
                            for i = 1:3
                            img4D(:,1:videoParams.wPx/2,i,1) = img2D(:,1:videoParams.wPx/2);
							img4D(:,1:videoParams.wPx/2,i,2) = img2D(:,1:videoParams.wPx/2);
                            end
                        else
                            for i = 1:3
                            img4D(:,videoParams.wPx/2+1:end,i,1) = img2D(:,videoParams.wPx/2+1:end);
							img4D(:,videoParams.wPx/2+1:end,i,3) = img2D(:,videoParams.wPx/2+1:end);
                            end
                        end
                        
%                         if oddParams.hasOddStepTask
                            img2D_odd(kEye) = lum2valFcn(videoParams.lumBg + GaborOdd{iEye});
                            if iEye == 1
                                for i = 1:3
                                img4D(:,1:videoParams.wPx/2,i,1+4) = img2D_odd(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,2+4) = img2D_odd(:,1:videoParams.wPx/2);
                                
                                img4D(:,1:videoParams.wPx/2,i,1+16) = img2D_odd(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,2+16) = img2D_odd(:,1:videoParams.wPx/2);
                                
                                img4D(:,1:videoParams.wPx/2,i,1+20) = img2D(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,2+20) = img2D(:,1:videoParams.wPx/2);
                                end
                            else
                                for i = 1:3
                                img4D(:,videoParams.wPx/2+1:end,i,1+4) = img2D_odd(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,3+4) = img2D_odd(:,videoParams.wPx/2+1:end);
                                
                                img4D(:,videoParams.wPx/2+1:end,i,1+16) = img2D(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,3+16) = img2D(:,videoParams.wPx/2+1:end);
                                
                                img4D(:,videoParams.wPx/2+1:end,i,1+20) = img2D_odd(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,3+20) = img2D_odd(:,videoParams.wPx/2+1:end);
                                end
                            end
%                         end
                        
						img2D(kEye) = lum2valFcn( videoParams.lumBg - Gabor{iEye} );
						
						if iEye == 1
                            for i = 1:3
                            img4D(:,1:videoParams.wPx/2,i,4) = img2D(:,1:videoParams.wPx/2);
							img4D(:,1:videoParams.wPx/2,i,3) = img2D(:,1:videoParams.wPx/2);
                            end
                        else
                            for i = 1:3
                            img4D(:,videoParams.wPx/2+1:end,i,4) = img2D(:,videoParams.wPx/2+1:end);
							img4D(:,videoParams.wPx/2+1:end,i,2) = img2D(:,videoParams.wPx/2+1:end);
                            end
                        end
                        
%                         if oddParams.hasOddStepTask
                            img2D_odd(kEye) = lum2valFcn( videoParams.lumBg - GaborOdd{iEye} );
                            if iEye == 1
                                for i = 1:3
                                img4D(:,1:videoParams.wPx/2,i,4+4) = img2D_odd(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,3+4) = img2D_odd(:,1:videoParams.wPx/2);
                                
                                img4D(:,1:videoParams.wPx/2,i,4+16) = img2D(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,3+16) = img2D(:,1:videoParams.wPx/2);
                                
                                img4D(:,1:videoParams.wPx/2,i,4+20) = img2D_odd(:,1:videoParams.wPx/2);
                                img4D(:,1:videoParams.wPx/2,i,3+20) = img2D_odd(:,1:videoParams.wPx/2);
                                end
                            else
                                for i = 1:3
                                img4D(:,videoParams.wPx/2+1:end,i,4+4) = img2D_odd(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,2+4) = img2D_odd(:,videoParams.wPx/2+1:end);
                                
                                img4D(:,videoParams.wPx/2+1:end,i,4+16) = img2D(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,2+16) = img2D(:,videoParams.wPx/2+1:end);
                                
                                img4D(:,videoParams.wPx/2+1:end,i,4+20) = img2D_odd(:,videoParams.wPx/2+1:end);
                                img4D(:,videoParams.wPx/2+1:end,i,2+20) = img2D_odd(:,videoParams.wPx/2+1:end);
                                end
                            end
%                 
               
                            
 
				end
            end

			% Add fixation marks ML
			fixType = GrabCellValue( parameters{iS}, 'Fix Point' );
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),1,:)=255;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),2,:)=255;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),3,:)=255;

            img4D(:,:,:,9) = img4D(:,:,:,1);
            img4D(:,:,:,10) = img4D(:,:,:,2);
            img4D(:,:,:,11) = img4D(:,:,:,3);
            img4D(:,:,:,12) = img4D(:,:,:,4);
            
%             img4D(videoParams.hPx/2-videoParams.ppdXY(1)*.25:videoParams.hPx/2+videoParams.ppdXY(1)*.25,videoParams.wPx/2-videoParams.ppdXY(1)*.25:videoParams.wPx/2+videoParams.ppdXY(1)*.25,1,9:12)=0;
%             img4D(videoParams.hPx/2-videoParams.ppdXY(1)*.25:videoParams.hPx/2+videoParams.ppdXY(1)*.25,videoParams.wPx/2-videoParams.ppdXY(1)*.25:videoParams.wPx/2+videoParams.ppdXY(1)*.25,2,9:12)=255;
%             img4D(videoParams.hPx/2-videoParams.ppdXY(1)*.25:videoParams.hPx/2+videoParams.ppdXY(1)*.25,videoParams.wPx/2-videoParams.ppdXY(1)*.25:videoParams.wPx/2+videoParams.ppdXY(1)*.25,3,9:12)=255;

            % first images without cue
            img4D(:,:,:,13) = img4D(:,:,:,1);
            img4D(:,:,:,14) = img4D(:,:,:,2);
            img4D(:,:,:,15) = img4D(:,:,:,3);
            img4D(:,:,:,16) = img4D(:,:,:,4);
            save tempvars;
            % add CUE ML RTA
            if currCond==1 || currCond==3
            %left cue
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2-videoParams.ppdXY(1)*1.1):round(videoParams.wPx/2-videoParams.ppdXY(1)*0.6),1:3,[1:12 13:16 17:24])=255;
            % add post cue (response)
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2-videoParams.ppdXY(1)*1.1):round(videoParams.wPx/2-videoParams.ppdXY(1)*0.6),1,9:12)=0;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2-videoParams.ppdXY(1)*1.1):round(videoParams.wPx/2-videoParams.ppdXY(1)*0.6),2,9:12)=255;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2-videoParams.ppdXY(1)*1.1):round(videoParams.wPx/2-videoParams.ppdXY(1)*0.6),3,9:12)=255;
            end
            if currCond==2 || currCond==3
            %right cue
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2+videoParams.ppdXY(1)*.6):round(videoParams.wPx/2+videoParams.ppdXY(1)*1.1),1:3,[1:12 13:16 7:24])=255;
            % add post cue (response)
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2+videoParams.ppdXY(1)*.6):round(videoParams.wPx/2+videoParams.ppdXY(1)*1.1),1,9:12)=0;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2+videoParams.ppdXY(1)*.6):round(videoParams.wPx/2+videoParams.ppdXY(1)*1.1),2,9:12)=255;
            img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.1):round(videoParams.hPx/2+videoParams.ppdXY(1)*.1),round(videoParams.wPx/2+videoParams.ppdXY(1)*.6):round(videoParams.wPx/2+videoParams.ppdXY(1)*1.1),3,9:12)=255;
            end
            % add feedback
            if ~isFirstTrial
                if correct
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),1,13:16)=0;
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),2,13:16)=255;
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),3,13:16)=0;
                else
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),1,13:16)=255;
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),2,13:16)=0;
                    img4D(round(videoParams.hPx/2-videoParams.ppdXY(1)*.25):round(videoParams.hPx/2+videoParams.ppdXY(1)*.25),round(videoParams.wPx/2-videoParams.ppdXY(1)*.25):round(videoParams.wPx/2+videoParams.ppdXY(1)*.25),3,13:16)=0;

                end
            end
          
            %RTA minimize the size of the stimulus
            
          a=find(max(img4D(:,:,1,12))-128);
          minX=min(a) ; 
          maxX=max(a);
          b=find(max(img4D(:,:,1,12)')-128);
          
          maxY=max(b);
          minY=videoParams.hPx/2-(maxY-videoParams.hPx/2);
          
          img4D=img4D([minY:maxY],[minX:maxX],:,:);
%             if bilateral
%                 img4D=img4D(:,:,1,:);
%             end

			imgSeq = zeros( [ timingParams.nFrameTrial, 1 ], 'int32' );
			% [ ++, +0, 0+, 00 ] or [ ++, +-, -+, -- ]
				if timingParams.preludeType == 0			% dynamic
					% 1 + 2*[on or off]1 + [on or off]2 = 4, 3, 2, or 1
					imgSeq(:) = 1 + 2 * ( mod( 0+stimParams.nFramePhi(1):timingParams.nFrameTrial-1+stimParams.nFramePhi(1), stimParams.nFrameCycle(1) ) < stimParams.dutyFrames(1) )...
					              +     ( mod( 0+stimParams.nFramePhi(2):timingParams.nFrameTrial-1+stimParams.nFramePhi(2), stimParams.nFrameCycle(2) ) < stimParams.dutyFrames(2) );
				else
	imgSeq(timingParams.nFramePrelude+1:timingParams.nFramePrelude+timingParams.nFrameCore) = 1 + 2 * ( mod( 0+stimParams.nFramePhi(1):timingParams.nFrameCore-1+stimParams.nFramePhi(1), stimParams.nFrameCycle(1) ) < stimParams.dutyFrames(1) )...
					              +     ( mod( 0+stimParams.nFramePhi(2):timingParams.nFrameCore-1+stimParams.nFramePhi(2), stimParams.nFrameCycle(2) ) < stimParams.dutyFrames(2) );
                end
                
%         if oddParams.hasOddStepTask
%             for i = 1:length(oddParams.oddSteps)
%                 if oddParams.oddSteps(i) % 1
%                     imgSeq(timingParams.nFramePrelude + (i-1)*timingParams.nFrameBin + 1 : ...
%                         timingParams.nFramePrelude + (i-1)*timingParams.nFrameBin + round(timingParams.nFrameBin/5)) = imgSeq(timingParams.nFramePrelude + (i-1)*timingParams.nFrameBin + 1 : ...
%                         timingParams.nFramePrelude + (i-1)*timingParams.nFrameBin + round(timingParams.nFrameBin/5)) + 4;
%                 end
%             end
%         end
%ML RTA
if ~isempty(overlap)
        imgSeq(overlap) = imgSeq(overlap) + 4;
end
if ~isempty(targetUnique)
    imgSeq(targetUnique) = imgSeq(targetUnique) + 16;
end
if ~isempty(nontargetUnique)
    imgSeq(nontargetUnique) = imgSeq(nontargetUnique) + 20;
end

% if targetPresent
%         imgSeq(targetStart:targetStart+16-1) = imgSeq(targetStart:targetStart+16-1) + 4;
% end
% if nontargetPresent
%         imgSeq(nontargetStart:nontargetStart+16-1) = imgSeq(nontargetStart:nontargetStart+16-1) + 4;
% end
%postlude increased to last 2 bins        
imgSeq(end-timingParams.nFramePrelude*2+1:end) = [imgSeq(end-timingParams.nFramePrelude+1:end) + 8 imgSeq(end-timingParams.nFramePrelude+1:end) + 8];
%prelude
        imgSeq(1:timingParams.nFramePrelude) = imgSeq(1:timingParams.nFramePrelude) + 12;

        
    imgSeq( [ false; diff(imgSeq)==0 ] ) = 0;		% get rid of repeats
			switch timingParams.preludeType
			case 1								% blank
				imgSeq(1) = nImg;									% set blank frames @ beginning of prelude & postlude
				imgSeq(timingParams.nFramePrelude+timingParams.nFrameCore+1) = nImg;
			case 2								% static
				imgSeq(1) = imgSeq(timingParams.nFramePrelude+1);		% move 1st frame of core to beginning of prelude
				imgSeq(timingParams.nFramePrelude+1) = 0;
            end
            save tempvariables
			assignin( 'base', 'output', { true, img4D, imgSeq } )			% put local var into global space as 'output'
		catch ME
			disp(ME.message)
			for iStack = numel(ME.stack):-1:1
				disp(ME.stack(iStack))
            end
                     save(fullfile('tempvars.mat'));
		  
			assignin( 'base', 'output', { false, zeros([1 1 1 1],'uint8'), 1 } )
        end
		return

        
        
        
		function val = lum2valFcn(lum)
			val = uint8( ( videoParams.lumRes - 1 ) / ( videoParams.lumMax - videoParams.lumMin ) * ( lum - videoParams.lumMin ) );
		end

		function setRectMask
			% assuming validation would have caught any out-of-bounds indices here
			[ hRect, wRect ] = size( Gabor{iEye} );
			yUL = floor( ( hPx - hRect ) / 2 );
			xUL = floor( ( wPx - wRect ) / 2 );
			kEye(:) = false;
			kEye(yUL+1:yUL+hRect,xUL+1:xUL+wRect) = true;
		end

%{
		sweepType	= GrabCellValue( parameters{iS}, 'Sweep Type' );
		isSwept		= ~strcmpi( sweepType,'Fixed' );
		sweepStart	= GrabCellValue( parameters{iS}, 'Sweep Start' );
		sweepEnd	   = GrabCellValue( parameters{iS}, 'Sweep End' );
%}

	end

	function ppdXY = getPixelsPerDeg( wPx, hPx, wCm, hCm, dCm )
		switch 1
		case 1	% inverse [width,height] of 1 pixel @ screen center
			ppdXY = 1 ./ ( atand( [ wCm, hCm ] ./ [ wPx, hPx ] / 2 / dCm ) * 2 );
		case 2	% total pixels / total degrees
			ppdXY = [ wPx, hPx ] ./ atand( [ wCm, hCm ] / 2 / dCm ) / 2;
		end
% 		if abs( 1 - ppdXY(2)/ppdXY(1) ) > 5e-2
% 			warning( 'rectangular pixels' )
% 		end
% 		ppdXY(:) = sqrt( prod( ppdXY ) );		% force square pixels with true area
	end

	function rV = GrabCellValue( cellArray, col1string, outputCol )
		% pull out the column-2 (or other column) value of cell array row 
		% where column-1 matches input string
		if nargin < 3
			outputCol = 2;
		end
		rV = cellArray{ strcmp( cellArray(:,1), col1string ), outputCol };
	end

	function rV = GetParamArray( aPartName, aParamName )
		% *** Sin Step Types not included in logic below? ***
		
		% For the given part and parameter name, return an array of values
		% corresponding to the steps in a sweep.  If the requested param is
		% not swept, the array will contain all the same values.

		% tSpatFreqSweepValues = GetParamArray( '1', 'Spat Freq (cpd)' );

		% Here's an example of sweep type specs...
		%
		% definitions{end-2} =
		% 	{
		% 		'Fixed'         'constant'   { }
		% 		'Contrast'      'increasing' { { '1' 'Contrast (%)' } { '2' 'Contrast (%)' } }
		% 		'Spat Freq'      'increasing' { { '1' 'Spat Freq (cpd)' } { '2' 'Spat Freq (cpd)' } }
		% 	}

		tNCStps    = GrabCellValue( timing, 'nmbCoreSteps' );
		tSweepType = GrabCellValue( parameters{iS}, 'Sweep Type' );

		% we need to construct a swept array if any of the {name,value} in definitions{iSweep}{:,3}

		sweepList = GrabCellValue( definitions{iSweep}, tSweepType, 3 );		% { {part,param}, {part,param} ... }
		
		% check for sweep
		% determine if any definitions{iSweep}{ iRow, { {part,param}... } } match arguments aPartName, aParamName
		partMatch  = ismember( aPartName,  cellfun( @(x)x{1}, sweepList, 'UniformOutput', false ) ); % will be false for "'Fixed' 'constant' {}"
		paramMatch = ismember( aParamName, cellfun( @(x)x{2}, sweepList, 'UniformOutput', false ) );
		if partMatch && paramMatch
			tSweepStart = GrabCellValue( parameters{iS}, 'Sweep Start' );
			tSweepEnd   = GrabCellValue( parameters{iS}, 'Sweep End' );
			if strcmpi( GrabCellValue( parameters{iS}, 'Step Type' ), 'Lin Stair' );
				rV = linspace( tSweepStart, tSweepEnd, tNCStps )';
			else
				rV = logspace( log10(tSweepStart), log10(tSweepEnd), tNCStps )';
			end
		else
			rV = repmat( GrabCellValue( parameters{eval(['i',aPartName])}, aParamName ), [ tNCStps, 1 ] );
		end

	end

end


