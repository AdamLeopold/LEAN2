\Generated by Cardinal Operations

Maximize
   1455.47 x_('A','B')22:45_f + 436.14 x_('A','B')22:45_l 
   + 1484.6 x_('C','B')7:40_f + 450.8 x_('C','B')7:40_l 
   + 1520.65 x_('B','C')12:25_f + 596.4400000000001 x_('B','C')12:25_l 
Subject To
 capacity_constraint_('A','B')22:45: 1.2 x_('A','B')22:45_f 
   + x_('A','B')22:45_l <= 187
 balance_constraint_('A','B')22:45: 4.92 x_('A','B')22:45_0 
   + 0.72 x_('A','B')22:45_f + 0.72 x_('A','B')22:45_l <= 159.14
 scale_constraint_('A','B')22:45_f: 2.671 x_('A','B')22:45_0 
   - 0.9 x_('A','B')22:45_f >= 0
 scale_constraint_('A','B')22:45_l: 1.303 x_('A','B')22:45_0 
   - 0.9 x_('A','B')22:45_l >= 0
 capacity_constraint_('C','B')7:40: 1.2 x_('C','B')7:40_f 
   + x_('C','B')7:40_l <= 187
 balance_constraint_('C','B')7:40: 1.72 x_('C','B')7:40_0 
   + 0.97 x_('C','B')7:40_f + 0.97 x_('C','B')7:40_l <= 26.89
 scale_constraint_('C','B')7:40_f: 2.16 x_('C','B')7:40_0 
   - 1.2 x_('C','B')7:40_f >= 0
 scale_constraint_('C','B')7:40_l: 1.127 x_('C','B')7:40_0 
   - 1.2 x_('C','B')7:40_l >= 0
 capacity_constraint_('B','C')12:25: 1.2 x_('B','C')12:25_f 
   + x_('B','C')12:25_l <= 187
 balance_constraint_('B','C')12:25: 3.7 x_('B','C')12:25_0 
   + 0.9 x_('B','C')12:25_f + 0.9 x_('B','C')12:25_l <= 39.79
 scale_constraint_('B','C')12:25_f: 3.079 x_('B','C')12:25_0 
   - 2 x_('B','C')12:25_f >= 0
 scale_constraint_('B','C')12:25_l: 1.652 x_('B','C')12:25_0 
   - 2 x_('B','C')12:25_l >= 0
END

