\Generated by Cardinal Operations

Maximize
   1669.65 x_('A','C')15:45_f + 506.95 x_('A','C')15:45_l 
   + 1638.05 x_('C','A')14:15_f + 531.5 x_('C','A')14:15_l 
Subject To
 capacity_constraint_('A','C')15:45: 1.2 x_('A','C')15:45_f 
   + x_('A','C')15:45_l <= 187
 balance_constraint_('A','C')15:45: 4.92 x_('A','C')15:45_0 
   + 0.72 x_('A','C')15:45_f + 0.72 x_('A','C')15:45_l <= 154.21
 scale_constraint_('A','C')15:45_f: 2.803 x_('A','C')15:45_0 
   - 0.9 x_('A','C')15:45_f >= 0
 scale_constraint_('A','C')15:45_l: 1.367 x_('A','C')15:45_0 
   - 0.9 x_('A','C')15:45_l >= 0
 capacity_constraint_('C','A')14:15: 1.2 x_('C','A')14:15_f 
   + x_('C','A')14:15_l <= 187
 balance_constraint_('C','A')14:15: 1.72 x_('C','A')14:15_0 
   + 0.97 x_('C','A')14:15_f + 0.97 x_('C','A')14:15_l <= 109.5
 scale_constraint_('C','A')14:15_f: 2.826 x_('C','A')14:15_0 
   - 1.2 x_('C','A')14:15_f >= 0
 scale_constraint_('C','A')14:15_l: 1.475 x_('C','A')14:15_0 
   - 1.2 x_('C','A')14:15_l >= 0
END

