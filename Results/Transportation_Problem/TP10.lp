\Generated by Cardinal Operations

Minimize
   543.756480860856 x_0_0 + 23.68527614176465 x_0_1 
   + 23.67638673077303 x_0_2 + 447.7514367867377 x_0_3 
   + 883.9151090405642 x_1_0 + 0.0497768476557696 x_1_1 
   + 0.0350986687216299 x_1_2 + 44.45588531711622 x_1_3 
   + 537.3456896658107 x_2_0 + 23.76927465907511 x_2_1 
   + 498.9565924946546 x_2_2 + 440.6073789043978 x_2_3 
   + 1791.493192397229 x_3_0 + 68.21633865655126 x_3_1 
   + 1432.483733965675 x_3_2 + 1527.763542546273 x_3_3 
Subject To
 demand_constraint: x_0_0 + x_1_0 + x_2_0 + x_3_0 >= 94
 x_0_1 + x_1_1 + x_2_1 + x_3_1 >= 39
 x_0_2 + x_1_2 + x_2_2 + x_3_2 >= 65
 x_0_3 + x_1_3 + x_2_3 + x_3_3 >= 435
 capacity_constraint: x_0_0 + x_0_1 + x_0_2 + x_0_3 <= 2531
 x_1_0 + x_1_1 + x_1_2 + x_1_3 <= 20
 x_2_0 + x_2_1 + x_2_2 + x_2_3 <= 210
 x_3_0 + x_3_1 + x_3_2 + x_3_3 <= 241
END

