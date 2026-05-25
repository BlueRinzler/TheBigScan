# The Big Scan

Tool created to find stocks that are in consolidation or have upward momentum. Industry rankings are assigned. 
1. Needs an API that can provide daily OHLCV data.  
2. Consolidation is found using modification of the TTM squeeze, a rollback period of 10 days with a 2 day grace. 
3. Momentum requires to be above the 10, 20 and 50 day SMA.
4. Output is 3 CVS files. 

.env folder is required.
