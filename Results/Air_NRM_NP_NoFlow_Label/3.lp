\Generated by Cardinal Operations

Maximize
   1484.6 x_CB7:40_f + 450.8 x_CB7:40_l + 886.96 x_CB10:45_f 
   + 263.88 x_CB10:45_l + 1510.9 x_CB14:15_f + 496.51 x_CB14:15_l 
   + 1583 x_CB16:55_f + 558.48 x_CB16:55_l 
Subject To
 capacity_constraint_CB7:40: 1.2 x_CB7:40_f + x_CB7:40_l <= 187
 balance_constraint_CB7:40: 1.72 x_CB7:40_0 + 0.97 x_CB7:40_f 
   + 0.97 x_CB7:40_l <= 26.89
 M_constraint_CB7:40_0: - 10000 y_CB7:40 + x_CB7:40_0 <= 0
 M_constraint_CB7:40_f: - 10000 y_CB7:40 + x_CB7:40_f <= 0
 M_constraint_CB7:40_l: - 10000 y_CB7:40 + x_CB7:40_l <= 0
 scale_constraint_CB7:40_f: 2.16 x_CB7:40_0 - 1.2 x_CB7:40_f >= 0
 scale_constraint_CB7:40_l: 1.127 x_CB7:40_0 - 1.2 x_CB7:40_l >= 0
 capacity_constraint_CB10:45: 1.2 x_CB10:45_f + x_CB10:45_l <= 187
 balance_constraint_CB10:45: 1.72 x_CB10:45_0 + 0.97 x_CB10:45_f 
   + 0.97 x_CB10:45_l <= 143.36
 M_constraint_CB10:45_0: - 10000 y_CB10:45 + x_CB10:45_0 <= 0
 M_constraint_CB10:45_f: - 10000 y_CB10:45 + x_CB10:45_f <= 0
 M_constraint_CB10:45_l: - 10000 y_CB10:45 + x_CB10:45_l <= 0
 scale_constraint_CB10:45_f: 1.916 x_CB10:45_0 - 1.2 x_CB10:45_f >= 0
 scale_constraint_CB10:45_l: x_CB10:45_0 - 1.2 x_CB10:45_l >= 0
 capacity_constraint_CB14:15: 1.2 x_CB14:15_f + x_CB14:15_l <= 187
 balance_constraint_CB14:15: 1.72 x_CB14:15_0 + 0.97 x_CB14:15_f 
   + 0.97 x_CB14:15_l <= 39.64
 M_constraint_CB14:15_0: - 10000 y_CB14:15 + x_CB14:15_0 <= 0
 M_constraint_CB14:15_f: - 10000 y_CB14:15 + x_CB14:15_f <= 0
 M_constraint_CB14:15_l: - 10000 y_CB14:15 + x_CB14:15_l <= 0
 scale_constraint_CB14:15_f: 2.826 x_CB14:15_0 - 1.2 x_CB14:15_f >= 0
 scale_constraint_CB14:15_l: 1.475 x_CB14:15_0 - 1.2 x_CB14:15_l >= 0
 capacity_constraint_CB16:55: 1.2 x_CB16:55_f + x_CB16:55_l <= 187
 balance_constraint_CB16:55: 1.72 x_CB16:55_0 + 0.97 x_CB16:55_f 
   + 0.97 x_CB16:55_l <= 4.4
 M_constraint_CB16:55_0: - 10000 y_CB16:55 + x_CB16:55_0 <= 0
 M_constraint_CB16:55_f: - 10000 y_CB16:55 + x_CB16:55_f <= 0
 M_constraint_CB16:55_l: - 10000 y_CB16:55 + x_CB16:55_l <= 0
 scale_constraint_CB16:55_f: 2.826 x_CB16:55_0 - 1.2 x_CB16:55_f >= 0
 scale_constraint_CB16:55_l: 1.475 x_CB16:55_0 - 1.2 x_CB16:55_l >= 0
 cardinality_constraint: y_CB7:40 + y_CB10:45 + y_CB14:15 + y_CB16:55 <= 3
Binaries
 y_CB7:40  y_CB10:45  y_CB14:15  y_CB16:55  
END

