\Generated by Cardinal Operations

Maximize
   1523 x_('C','B')18:30_f + 443.91 x_('C','B')18:30_l 
   + 1665.21 x_('C','A')18:30_f + 463.64 x_('C','A')18:30_l 
Subject To
 capacity_constraint_('C','B')18:30: 1.2 x_('C','B')18:30_f 
   + x_('C','B')18:30_l <= 187
 balance_constraint_('C','B')18:30: 1.72 x_('C','B')18:30_0 
   + 0.97 x_('C','B')18:30_f + 0.97 x_('C','B')18:30_l <= 16.95
 scale_constraint_('C','B')18:30_f: 3.126 x_('C','B')18:30_0 
   - 1.2 x_('C','B')18:30_f >= 0
 scale_constraint_('C','B')18:30_l: 1.631 x_('C','B')18:30_0 
   - 1.2 x_('C','B')18:30_l >= 0
 capacity_constraint_('C','A')18:30: 1.2 x_('C','A')18:30_f 
   + x_('C','A')18:30_l <= 187
 balance_constraint_('C','A')18:30: 1.72 x_('C','A')18:30_0 
   + 0.97 x_('C','A')18:30_f + 0.97 x_('C','A')18:30_l <= 166.5
 scale_constraint_('C','A')18:30_f: 3.126 x_('C','A')18:30_0 
   - 1.2 x_('C','A')18:30_f >= 0
 scale_constraint_('C','A')18:30_l: 1.631 x_('C','A')18:30_0 
   - 1.2 x_('C','A')18:30_l >= 0
END

