\Generated by Cardinal Operations

Maximize
   1638.05 x_CA14:15_f + 531.5 x_CA14:15_l + 1665.33 x_CA16:55_f 
   + 493.28 x_CA16:55_l + 1665.21 x_CA18:30_f + 463.64 x_CA18:30_l 
   + 1656.11 x_CA7:40_f + 476.24 x_CA7:40_l 
Subject To
 capacity_constraint_CA14:15: 1.2 x_CA14:15_f + x_CA14:15_l <= 187
 balance_constraint_CA14:15: 1.72 x_CA14:15_0 + 0.97 x_CA14:15_f 
   + 0.97 x_CA14:15_l <= 109.5
 M_constraint_CA14:15_0: - 10000 y_CA14:15 + x_CA14:15_0 <= 0
 M_constraint_CA14:15_f: - 10000 y_CA14:15 + x_CA14:15_f <= 0
 M_constraint_CA14:15_l: - 10000 y_CA14:15 + x_CA14:15_l <= 0
 scale_constraint_CA14:15_f: 2.826 x_CA14:15_0 - 1.2 x_CA14:15_f >= 0
 scale_constraint_CA14:15_l: 1.475 x_CA14:15_0 - 1.2 x_CA14:15_l >= 0
 capacity_constraint_CA16:55: 1.2 x_CA16:55_f + x_CA16:55_l <= 187
 balance_constraint_CA16:55: 1.72 x_CA16:55_0 + 0.97 x_CA16:55_f 
   + 0.97 x_CA16:55_l <= 173.79
 M_constraint_CA16:55_0: - 10000 y_CA16:55 + x_CA16:55_0 <= 0
 M_constraint_CA16:55_f: - 10000 y_CA16:55 + x_CA16:55_f <= 0
 M_constraint_CA16:55_l: - 10000 y_CA16:55 + x_CA16:55_l <= 0
 scale_constraint_CA16:55_f: 2.826 x_CA16:55_0 - 1.2 x_CA16:55_f >= 0
 scale_constraint_CA16:55_l: 1.475 x_CA16:55_0 - 1.2 x_CA16:55_l >= 0
 capacity_constraint_CA18:30: 1.2 x_CA18:30_f + x_CA18:30_l <= 187
 balance_constraint_CA18:30: 1.72 x_CA18:30_0 + 0.97 x_CA18:30_f 
   + 0.97 x_CA18:30_l <= 166.5
 M_constraint_CA18:30_0: - 10000 y_CA18:30 + x_CA18:30_0 <= 0
 M_constraint_CA18:30_f: - 10000 y_CA18:30 + x_CA18:30_f <= 0
 M_constraint_CA18:30_l: - 10000 y_CA18:30 + x_CA18:30_l <= 0
 scale_constraint_CA18:30_f: 3.126 x_CA18:30_0 - 1.2 x_CA18:30_f >= 0
 scale_constraint_CA18:30_l: 1.631 x_CA18:30_0 - 1.2 x_CA18:30_l >= 0
 capacity_constraint_CA7:40: 1.2 x_CA7:40_f + x_CA7:40_l <= 187
 balance_constraint_CA7:40: 1.72 x_CA7:40_0 + 0.97 x_CA7:40_f 
   + 0.97 x_CA7:40_l <= 158.71
 M_constraint_CA7:40_0: - 10000 y_CA7:40 + x_CA7:40_0 <= 0
 M_constraint_CA7:40_f: - 10000 y_CA7:40 + x_CA7:40_f <= 0
 M_constraint_CA7:40_l: - 10000 y_CA7:40 + x_CA7:40_l <= 0
 scale_constraint_CA7:40_f: 2.16 x_CA7:40_0 - 1.2 x_CA7:40_f >= 0
 scale_constraint_CA7:40_l: 1.127 x_CA7:40_0 - 1.2 x_CA7:40_l >= 0
 cardinality_constraint: y_CA14:15 + y_CA16:55 + y_CA18:30 + y_CA7:40 <= 3
Binaries
 y_CA14:15  y_CA16:55  y_CA18:30  y_CA7:40  
END

