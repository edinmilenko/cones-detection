===== TRAINING 9-stage =====
<BEGIN
POS count : consumed   12500 : 13080
^C^C^Crrent samples: 898
got 3 SIGTERM/SIGINTs, forcefully exiting

the process might get stuck cuz it removed negative samples till having no more than 989 (in this case), making it loop infinitely searching other 7500-898 in a empty search space.

solutions are either making the 7500 window bigger or limit the process to 8 steps (in this case, as the last that completed successfully is the 8th one).