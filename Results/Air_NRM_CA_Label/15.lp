\Generated by Cardinal Operations

Maximize
   1464.07 x_('A','B')11:20_f + 441.88 x_('A','B')11:20_l 
   + 1544.13 x_('B','C')15:40_f + 483.3 x_('B','C')15:40_l 
   + 1484.6 x_('C','B')7:40_f + 450.8 x_('C','B')7:40_l 
Subject To
 capacity_constraint_('A','B')11:20: 1.2 x_('A','B')11:20_f 
   + x_('A','B')11:20_l <= 187
 balance_constraint_('A','B')11:20: 4.92 x_('A','B')11:20_0 
   + 0.72 x_('A','B')11:20_f + 0.72 x_('A','B')11:20_l <= 179.35
 scale_constraint_('A','B')11:20_f: 2.05 x_('A','B')11:20_0 
   - 0.9 x_('A','B')11:20_f >= 0
 scale_constraint_('A','B')11:20_l: x_('A','B')11:20_0 
   - 0.9 x_('A','B')11:20_l >= 0
 capacity_constraint_('B','C')15:40: 1.2 x_('B','C')15:40_f 
   + x_('B','C')15:40_l <= 187
 balance_constraint_('B','C')15:40: 3.7 x_('B','C')15:40_0 
   + 0.9 x_('B','C')15:40_f + 0.9 x_('B','C')15:40_l <= 14.96
 scale_constraint_('B','C')15:40_f: 3.079 x_('B','C')15:40_0 
   - 2 x_('B','C')15:40_f >= 0
 scale_constraint_('B','C')15:40_l: 1.652 x_('B','C')15:40_0 
   - 2 x_('B','C')15:40_l >= 0
 capacity_constraint_('C','B')7:40: 1.2 x_('C','B')7:40_f 
   + x_('C','B')7:40_l <= 187
 balance_constraint_('C','B')7:40: 1.72 x_('C','B')7:40_0 
   + 0.97 x_('C','B')7:40_f + 0.97 x_('C','B')7:40_l <= 26.89
 scale_constraint_('C','B')7:40_f: 2.16 x_('C','B')7:40_0 
   - 1.2 x_('C','B')7:40_f >= 0
 scale_constraint_('C','B')7:40_l: 1.127 x_('C','B')7:40_0 
   - 1.2 x_('C','B')7:40_l >= 0
END

