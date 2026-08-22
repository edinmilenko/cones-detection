scaleFactor: this parameter should be in [1.05, 1.2] range. higher values like 1.4 help reducing false positives, but it means each img gets reduced by 40% each pass. tradeoff is we destroy recall, as detector will fail systematically on mid-distance cones as their distance in pixel "skips" the resolutions of the training window.

false positives are used for training ~ hard negative mining, without reducing searching space

minNeighbors could have its best value in (8, 12, 15, 18), this needs to be tuned better though